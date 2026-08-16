"""Phase 2 — MC Dropout sur un checkpoint "_dropout" donné. Génère les cartes
d'incertitude et valide qualitativement + quantitativement (corrélation
incertitude/erreur, incertitude moyenne par structure) avant d'en tirer des
conclusions.

Usage: python scripts/run_mc_dropout.py --sequence FLASH --fold 1
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import pointbiserialr

from src.dataset import TOSDataset
from src.metrics import STRUCTURE_NAMES
from src.model import UNet2D
from src.splitting import get_kfold_splits
from src.visualization import plot_uncertainty
from src.xai import mc_dropout_predict, predictive_entropy

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "DATA_processed"
OUT_DIR = ROOT / "report" / "figures"
NEEDS_REVIEW = ["2017-110_01-0224-V1MR"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=["FLASH", "RSSFP"])
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    tag_prefix = f"xai_mcdropout_{args.sequence}_fold{args.fold}"

    device = get_device()
    print(f"Device: {device}")

    ckpt_path = ROOT / "checkpoints" / f"checkpoint_{args.sequence}_fold{args.fold}_dropout.pt"
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = UNet2D(in_channels=1, n_classes=7, base_ch=32, dropout=checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Checkpoint chargé : {ckpt_path.name} -- fold {checkpoint['fold']}, "
          f"séquence {checkpoint['sequence']}, dropout {checkpoint['dropout']}, epoch {checkpoint['epoch']}")

    splits = get_kfold_splits(PROCESSED / "manifest.csv", k=3, always_train=NEEDS_REVIEW)
    val_patients = splits[args.fold]["val"]
    val_ds = TOSDataset(PROCESSED, val_patients, crop=False, sequence=args.sequence)

    rng = random.Random(0)  # même graine que run_xai.py -- mêmes échantillons, comparables entre modèles
    sample_idxs = rng.sample(range(len(val_ds)), 4)

    all_entropy, all_correct, all_structure_entropy = [], [], {name: [] for name in STRUCTURE_NAMES[1:]}

    print("\n=== MC Dropout (30 passes) : 4 échantillons ===")
    for i, idx in enumerate(sample_idxs):
        image, label = val_ds[idx]
        image_batch = image.unsqueeze(0).to(device)
        label_np = label.numpy()
        img_np = image.squeeze(0).numpy()

        mean_probs, std_probs = mc_dropout_predict(model, image_batch, n_samples=30)
        pred = mean_probs.argmax(dim=1).squeeze(0).cpu().numpy()
        entropy = predictive_entropy(mean_probs)
        error_mask = (pred != label_np).astype(float)

        out_path = OUT_DIR / f"{tag_prefix}_sample{i}.png"
        plot_uncertainty(img_np, entropy, label_np, pred, error_mask, out_path)

        all_entropy.append(entropy.flatten())
        all_correct.append((pred == label_np).astype(int).flatten())
        for class_idx in range(1, 7):
            struct_mask = (label_np == class_idx) | (pred == class_idx)
            if struct_mask.any():
                all_structure_entropy[STRUCTURE_NAMES[class_idx]].append(float(entropy[struct_mask].mean()))

        print(f"  échantillon {i} (idx={idx}) : entropie moyenne={entropy.mean():.4f}, "
              f"erreur={error_mask.mean()*100:.2f}% des pixels")

    print("\n=== Validation quantitative ===")
    entropy_all = np.concatenate(all_entropy)
    correct_all = np.concatenate(all_correct)
    corr, pval = pointbiserialr(correct_all, entropy_all)
    print(f"Corrélation point-bisériale (correction, entropie) sur les 4 échantillons "
          f"({len(entropy_all)} pixels) : r={corr:.4f}, p={pval:.2e}")
    print(f"Entropie moyenne sur pixels corrects   : {entropy_all[correct_all==1].mean():.4f}")
    print(f"Entropie moyenne sur pixels en erreur   : {entropy_all[correct_all==0].mean():.4f}")

    print("\nEntropie moyenne par structure (sur les pixels GT ou prédits, 4 échantillons) :")
    for name, values in all_structure_entropy.items():
        if values:
            print(f"  {name:6s}  entropie moyenne = {np.mean(values):.4f}  (n={len(values)} échantillons)")

    print(f"\nFigures sauvegardées dans {OUT_DIR}")


if __name__ == "__main__":
    main()
