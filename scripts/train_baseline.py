"""Baseline FLASH/RSSFP — entraîne un modèle spécialisé par type de séquence
(comparaison H2 au niveau modèle, pas seulement au niveau contraste brut).

Méthodologie fixée à l'avance à partir de la littérature (pas d'ablation empirique
aug/crop comme dans scripts/train_phase1.py) :
  - Augmentation TOUJOURS active (nnU-Net, Isensee et al. 2021, l'applique par défaut
    sur toutes les tâches — ce n'est pas une option à tester).
  - Recadrage ROI JAMAIS actif (un crop fixe non appris n'est pas une pratique
    standard ; nnU-Net ne fait pas de crop fixe par défaut).
  - Perte Dice + CE pondérée (Ma et al. 2021, robuste sur déséquilibre modéré).
  - Métriques Dice + Hausdorff-95 + NSD (Maier-Hein et al. 2024, "Metrics reloaded").
  - k=3 folds par défaut (voir src/splitting.py) ; comparaison finale FLASH vs RSSFP
    au niveau PATIENT (pas au niveau fold), voir evaluate_per_patient ci-dessous.

Usage: python scripts/train_baseline.py --sequence FLASH --fold 0 --epochs 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.augmentation import JointAugmentation
from src.dataset import TOSDataset
from src.losses import DiceCELoss, compute_class_weights
from src.metrics import (
    dice_per_class, hausdorff95_per_class, nsd_per_class,
    precision_recall_per_class, STRUCTURE_NAMES,
)
from src.model import UNet2D
from src.splitting import get_kfold_splits
from src.visualization import generate_all_figures

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "DATA_processed"
NEEDS_REVIEW = ["2017-110_01-0224-V1MR"]
PER_PATIENT_CSV = ROOT / "report" / "figures" / "baseline_per_patient_metrics.csv"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_fast(model, loader, device, n_classes=7):
    """Dice seul (pas HD95/NSD, trop coûteux) — suivi à chaque epoch pour choisir le
    meilleur checkpoint, pas pour le résultat final rapporté."""
    model.eval()
    total_dice = torch.zeros(n_classes)
    n_batches = 0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image)
            total_dice += dice_per_class(logits, label, n_classes).cpu()
            n_batches += 1
    return total_dice / n_batches


def evaluate_per_patient(model, val_patients, sequence, fold, device, n_classes=7, batch_size=8):
    """Évalue le meilleur checkpoint patient par patient (pas moyenné sur le fold) --
    nécessaire pour le test de Wilcoxon apparié FLASH vs RSSFP au niveau patient
    (chaque patient valide exactement une fois par modèle sur les k folds, donc c'est
    le niveau qui donne le plus de paires exploitables, pas le fold — voir discussion
    méthodologique). Renvoie une liste de lignes (une par patient x structure)."""
    model.eval()
    rows = []
    for patient in val_patients:
        try:
            ds = TOSDataset(PROCESSED, [patient], sequence=sequence, crop=False)
        except ValueError as e:
            print(f"  [attention] patient {patient} ignoré pour sequence={sequence} : {e}")
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        total_dice = torch.zeros(n_classes)
        total_precision = torch.zeros(n_classes)
        total_recall = torch.zeros(n_classes)
        total_hd = torch.zeros(n_classes)
        hd_counts = torch.zeros(n_classes)
        total_nsd = torch.zeros(n_classes)
        nsd_counts = torch.zeros(n_classes)
        n_batches = 0
        with torch.no_grad():
            for image, label in loader:
                image, label = image.to(device), label.to(device)
                logits = model(image)
                total_dice += dice_per_class(logits, label, n_classes).cpu()
                precision, recall = precision_recall_per_class(logits, label, n_classes)
                total_precision += precision.cpu()
                total_recall += recall.cpu()
                batch_hd = hausdorff95_per_class(logits, label, n_classes)
                valid = ~torch.isnan(batch_hd)
                total_hd[valid] += batch_hd[valid]
                hd_counts[valid] += 1
                batch_nsd = nsd_per_class(logits, label, n_classes)
                validn = ~torch.isnan(batch_nsd)
                total_nsd[validn] += batch_nsd[validn]
                nsd_counts[validn] += 1
                n_batches += 1

        dice = total_dice / n_batches
        precision = total_precision / n_batches
        recall = total_recall / n_batches
        hd95 = total_hd / hd_counts.clamp(min=1)
        hd95[hd_counts == 0] = float("nan")
        nsd = total_nsd / nsd_counts.clamp(min=1)
        nsd[nsd_counts == 0] = float("nan")

        for c in range(n_classes):
            rows.append({
                "patient": patient, "sequence": sequence, "fold": fold,
                "structure": STRUCTURE_NAMES[c],
                "dice": dice[c].item(), "precision": precision[c].item(),
                "recall": recall[c].item(), "hausdorff95": hd95[c].item(),
                "nsd": nsd[c].item(),
            })
    return rows


def save_per_patient_rows(rows):
    """Ajoute les lignes au CSV persistant, en dédupliquant par (patient, sequence,
    structure) -- évite les doublons si un run est relancé (leçon de l'incident du
    checkpoint écrasé, voir mémoire de session : ne pas silencieusement accumuler des
    résultats obsolètes à côté des nouveaux)."""
    new_df = pd.DataFrame(rows)
    PER_PATIENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if PER_PATIENT_CSV.exists():
        old_df = pd.read_csv(PER_PATIENT_CSV)
        key_cols = ["patient", "sequence", "structure"]
        old_df = old_df[~old_df.set_index(key_cols).index.isin(new_df.set_index(key_cols).index)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(PER_PATIENT_CSV, index=False)
    print(f"  {len(new_df)} lignes ajoutées à {PER_PATIENT_CSV} ({len(combined)} au total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=["FLASH", "RSSFP"],
                         help="Modèle spécialisé par séquence -- pas de modèle pooled dans ce baseline")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0,
                         help="Dropout2d (bottleneck/enc4/dec4 uniquement, Kendall et al. 2015) pour "
                              "MC Dropout -- Phase 2 XAI. 0.0 par défaut = architecture identique aux "
                              "6 checkpoints du baseline Phase 1. Produit un checkpoint suffixé "
                              "'_dropout', jamais celui du baseline (pas de comparaison Phase 1 faussée).")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Séquence: {args.sequence} (augmentation: activée, crop ROI: inactif -- fixé par méthodologie, "
          f"dropout: {args.dropout})")

    splits = get_kfold_splits(PROCESSED / "manifest.csv", k=args.k, always_train=NEEDS_REVIEW)
    train_patients, val_patients = splits[args.fold]["train"], splits[args.fold]["val"]
    print(f"Fold {args.fold}/{args.k} — train: {len(train_patients)} patients, val: {len(val_patients)} patients")
    print(f"  val = {val_patients}")

    transform = JointAugmentation()
    train_ds = TOSDataset(PROCESSED, train_patients, transform=transform, crop=False, sequence=args.sequence)
    val_ds = TOSDataset(PROCESSED, val_patients, crop=False, sequence=args.sequence)  # jamais d'augmentation sur la validation
    print(f"  {len(train_ds)} frames train, {len(val_ds)} frames val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_diag_ds = TOSDataset(PROCESSED, train_patients, crop=False, sequence=args.sequence)
    rng = random.Random(0)
    diag_idx = rng.sample(range(len(train_diag_ds)), min(len(val_ds), len(train_diag_ds)))
    train_diag_loader = DataLoader(Subset(train_diag_ds, diag_idx), batch_size=args.batch_size, shuffle=False)

    print("Calcul des poids de classe (fréquence inverse, mesurée sur le train set)...")
    class_weights = compute_class_weights(train_ds, n_classes=7).to(device)
    print("  poids:", {name: round(w, 2) for name, w in zip(STRUCTURE_NAMES, class_weights.tolist())})

    model = UNet2D(in_channels=1, n_classes=7, base_ch=32, dropout=args.dropout).to(device)
    criterion = DiceCELoss(n_classes=7, class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mean_dice = -1.0
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    suffix = "_dropout" if args.dropout > 0 else ""
    ckpt_path = ckpt_dir / f"checkpoint_{args.sequence}_fold{args.fold}{suffix}.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_dice = evaluate_fast(model, train_diag_loader, device)
        val_dice = evaluate_fast(model, val_loader, device)
        mean_fg_dice_train = train_dice[1:].mean().item()
        mean_fg_dice = val_dice[1:].mean().item()
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"dice_train={mean_fg_dice_train:.4f}  dice_val={mean_fg_dice:.4f}  ({time.time()-t0:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "train_dice_per_class": train_dice.tolist(), "val_dice_per_class": val_dice.tolist()})

        if mean_fg_dice > best_mean_dice:
            best_mean_dice = mean_fg_dice
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_dice_per_class": val_dice, "fold": args.fold,
                        "sequence": args.sequence, "dropout": args.dropout}, ckpt_path)

    print(f"\nMeilleur Dice moyen (structures) : {best_mean_dice:.4f} — checkpoint : {ckpt_path}")
    print("Évaluation finale du meilleur checkpoint, PAR PATIENT (Dice + Précision + Rappel + HD95 + NSD)...")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    rows = evaluate_per_patient(model, val_patients, args.sequence, args.fold, device)
    if args.dropout > 0:
        print("  dropout>0 : run non ajouté à baseline_per_patient_metrics.csv "
              "(réservé aux 6 runs Phase 1 sans dropout, comparaison FLASH/RSSFP déjà close)")
    else:
        save_per_patient_rows(rows)

    per_patient_df = pd.DataFrame(rows)
    pooled = per_patient_df.groupby("structure")[["dice", "precision", "recall", "hausdorff95", "nsd"]].mean()
    pooled = pooled.reindex(STRUCTURE_NAMES)  # ordre canonique fond..PB

    print(f"\nDice / Précision / Rappel / Hausdorff-95 / NSD par structure "
          f"(moyenne inter-patients, meilleur checkpoint, epoch {checkpoint['epoch']}) :")
    for name in STRUCTURE_NAMES:
        r = pooled.loc[name]
        print(f"  {name:6s}  dice={r.dice:.4f}  precision={r.precision:.4f}  rappel={r.recall:.4f}  "
              f"hausdorff95={r.hausdorff95:.2f}px  nsd={r.nsd:.4f}")

    print("\nGénération des courbes et figures...")
    fig_dir = ROOT / "report" / "figures"
    final_metrics = {
        "dice": torch.tensor(pooled["dice"].values, dtype=torch.float32),
        "precision": torch.tensor(pooled["precision"].values, dtype=torch.float32),
        "recall": torch.tensor(pooled["recall"].values, dtype=torch.float32),
        "hausdorff95": torch.tensor(pooled["hausdorff95"].values, dtype=torch.float32),
    }
    tag = f"baseline_{args.sequence}_fold{args.fold}{suffix}"
    generate_all_figures(history, final_metrics, model, val_ds, device, fig_dir, tag)


if __name__ == "__main__":
    main()
