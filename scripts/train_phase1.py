"""Phase 1 — entraîne le U-Net 2D baseline sur un fold donné (validation croisée
GroupKFold au niveau patient, voir src/splitting.py). Dice/Hausdorff par structure
comme référence de sanité, pas comme objectif d'optimisation (voir chapitre Contexte
du rapport : l'interprétabilité est la contribution centrale de la thèse).

Usage: python scripts/train_phase1.py --fold 0 --epochs 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import torch
from torch.utils.data import DataLoader, Subset

from src.augmentation import JointAugmentation
from src.dataset import TOSDataset
from src.losses import DiceCELoss, compute_class_weights
from src.metrics import dice_per_class, hausdorff95_per_class, precision_recall_per_class, STRUCTURE_NAMES
from src.model import UNet2D
from src.splitting import get_kfold_splits
from src.visualization import generate_all_figures

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "DATA_processed"
NEEDS_REVIEW = ["2017-110_01-0224-V1MR"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device, n_classes=7, with_hausdorff=False):
    """Dice + precision/rappel a chaque appel (peu couteux, meme cout que le Dice).
    Hausdorff-95 seulement si demande (with_hausdorff) — plus couteux (distance
    transform par image), reserve a l'evaluation finale du meilleur checkpoint, pas a
    chaque epoch."""
    model.eval()
    total_dice = torch.zeros(n_classes)
    total_precision = torch.zeros(n_classes)
    total_recall = torch.zeros(n_classes)
    total_hd = torch.zeros(n_classes)
    hd_counts = torch.zeros(n_classes)
    n_batches = 0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image)
            total_dice += dice_per_class(logits, label, n_classes).cpu()
            precision, recall = precision_recall_per_class(logits, label, n_classes)
            total_precision += precision.cpu()
            total_recall += recall.cpu()
            if with_hausdorff:
                batch_hd = hausdorff95_per_class(logits, label, n_classes)
                valid = ~torch.isnan(batch_hd)
                total_hd[valid] += batch_hd[valid]
                hd_counts[valid] += 1
            n_batches += 1
    result = {
        "dice": total_dice / n_batches,
        "precision": total_precision / n_batches,
        "recall": total_recall / n_batches,
    }
    if with_hausdorff:
        result["hausdorff95"] = total_hd / hd_counts.clamp(min=1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true",
                         help="Active l'augmentation de données (train uniquement, jamais val) — voir src/augmentation.py")
    parser.add_argument("--no-crop", action="store_true",
                         help="Désactive le recadrage ROI fixe (actif par défaut) — voir ROI_CROP dans src/dataset.py")
    args = parser.parse_args()
    crop = not args.no_crop
    tag = ("aug" if args.augment else "noaug") + ("_crop" if crop else "_nocrop")

    device = get_device()
    print(f"Device: {device}")

    splits = get_kfold_splits(PROCESSED / "manifest.csv", k=args.k, always_train=NEEDS_REVIEW)
    train_patients, val_patients = splits[args.fold]["train"], splits[args.fold]["val"]
    print(f"Fold {args.fold}/{args.k} — train: {len(train_patients)} patients, val: {len(val_patients)} patients")
    print(f"  val = {val_patients}")

    transform = JointAugmentation() if args.augment else None
    train_ds = TOSDataset(PROCESSED, train_patients, transform=transform, crop=crop)
    val_ds = TOSDataset(PROCESSED, val_patients, crop=crop)  # jamais d'augmentation sur la validation
    print(f"  {len(train_ds)} frames train, {len(val_ds)} frames val  "
          f"(augmentation: {'activee' if args.augment else 'desactivee'}, crop ROI: {'actif' if crop else 'inactif'})")
    if crop:
        print(f"  taille apres crop : {train_ds[0][0].shape[-2:]}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # sous-echantillon fixe du train set (non augmente), meme taille que le val set,
    # pour suivre le Dice train a cout raisonnable et diagnostiquer le sur-apprentissage
    # (comparer a val_dice) -- evaluer sur les 780 frames completes chaque epoch doublerait
    # le temps d'epoch, deja lent sur MPS.
    train_diag_ds = TOSDataset(PROCESSED, train_patients, crop=crop)  # sans transform, pour un Dice comparable
    rng = random.Random(0)
    diag_idx = rng.sample(range(len(train_diag_ds)), min(len(val_ds), len(train_diag_ds)))
    train_diag_loader = DataLoader(Subset(train_diag_ds, diag_idx), batch_size=args.batch_size, shuffle=False)

    print("Calcul des poids de classe (fréquence inverse, mesurée sur le train set)...")
    class_weights = compute_class_weights(train_ds, n_classes=7).to(device)
    print("  poids:", {name: round(w, 2) for name, w in zip(STRUCTURE_NAMES, class_weights.tolist())})

    model = UNet2D(in_channels=1, n_classes=7, base_ch=32).to(device)
    criterion = DiceCELoss(n_classes=7, class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mean_dice = -1.0
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_fold{args.fold}_{tag}.pt"
    history = []  # trace d'entraînement, une ligne par epoch -- sert aux courbes generees a la fin

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
        train_metrics = evaluate(model, train_diag_loader, device)  # sous-echantillon fixe, non augmente
        val_metrics = evaluate(model, val_loader, device)  # Dice seul, pas Hausdorff (trop lent a chaque epoch)
        train_dice, val_dice = train_metrics["dice"], val_metrics["dice"]
        mean_fg_dice_train = train_dice[1:].mean().item()
        mean_fg_dice = val_dice[1:].mean().item()  # moyenne sur les 6 structures, hors fond
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"dice_train={mean_fg_dice_train:.4f}  dice_val={mean_fg_dice:.4f}  ({time.time()-t0:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "train_dice_per_class": train_dice.tolist(), "val_dice_per_class": val_dice.tolist()})

        if mean_fg_dice > best_mean_dice:
            best_mean_dice = mean_fg_dice
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_dice_per_class": val_dice, "fold": args.fold,
                        "augment": args.augment, "crop": crop}, ckpt_path)

    print(f"\nMeilleur Dice moyen (structures) : {best_mean_dice:.4f} — checkpoint : {ckpt_path}")
    print("Évaluation finale du meilleur checkpoint (Dice + Hausdorff-95)...")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = evaluate(model, val_loader, device, with_hausdorff=True)

    print("\nDice / Précision / Rappel / Hausdorff-95 par structure (meilleur checkpoint, epoch %d) :" % checkpoint["epoch"])
    for name, d, p, r, hd in zip(
        STRUCTURE_NAMES, final_metrics["dice"].tolist(), final_metrics["precision"].tolist(),
        final_metrics["recall"].tolist(), final_metrics["hausdorff95"].tolist(),
    ):
        hd_str = f"{hd:.2f}px" if not (hd != hd) else "n/a (classe absente du val set)"
        print(f"  {name:6s}  dice={d:.4f}  precision={p:.4f}  rappel={r:.4f}  hausdorff95={hd_str}")

    print("\nGénération des courbes et figures...")
    fig_dir = ROOT / "report" / "figures"
    generate_all_figures(history, final_metrics, model, val_ds, device, fig_dir, f"{args.fold}_{tag}")


if __name__ == "__main__":
    main()
