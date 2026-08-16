"""Phase 3 — analyse temporelle sur une série complète (courbe de compression
costoclaviculaire, incertitude MC Dropout, cohérence temporelle de Seg-Grad-CAM).

Fonctionne sur autant de frames que présentes dans DATA/<patient>/<série>/ -- 20
actuellement (pilote/validation de code), jusqu'à ~300 une fois la séquence complète
récupérée, sans changement de code (voir src/phase3.py, load_full_sequence).

Usage: python scripts/run_phase3_temporal.py --patient "2017-110 01-0230-V1MR" --serie "FLASH D"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.metrics import STRUCTURE_NAMES
from src.model import UNet2D
from src.phase3 import (
    costoclavicular_distance, load_full_sequence, predict_sequence,
    temporal_saliency_consistency,
)
from src.visualization import plot_phase3_dashboard
from src.xai import SegGradCAM, mc_dropout_predict, predictive_entropy

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DATA"
OUT_DIR = ROOT / "report" / "figures"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--serie", type=str, required=True, help="ex. 'FLASH D'")
    parser.add_argument("--sequence", type=str, default="FLASH", choices=["FLASH", "RSSFP"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k1-class", type=int, default=5)
    parser.add_argument("--data-root", type=str, default=str(DATA),
                         help="Racine des données DICOM brutes -- DATA/ (20 frames, défaut) ou "
                              "big_dataset/ (séquence complète, une fois récupérée)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    series_dir = Path(args.data_root) / args.patient / args.serie
    print(f"Chargement de {series_dir}...")
    image, instance_numbers = load_full_sequence(series_dir)
    n_frames = image.shape[-1]
    print(f"{n_frames} frames chargées (InstanceNumber {instance_numbers[0]}--{instance_numbers[-1]})")
    if n_frames < 20 + 1:
        print("  [pilote] séquence complète pas encore récupérée -- ceci valide le code, "
              "pas encore la vraie analyse Phase 3 (voir rapport, limite assumée).")

    ckpt_plain = torch.load(ROOT / "checkpoints" / f"checkpoint_{args.sequence}_fold{args.fold}.pt",
                             weights_only=False, map_location=device)
    model = UNet2D(in_channels=1, n_classes=7, base_ch=32).to(device)
    model.load_state_dict(ckpt_plain["model_state"])
    model.eval()

    ckpt_drop = torch.load(ROOT / "checkpoints" / f"checkpoint_{args.sequence}_fold{args.fold}_dropout.pt",
                            weights_only=False, map_location=device)
    model_drop = UNet2D(in_channels=1, n_classes=7, base_ch=32, dropout=ckpt_drop["dropout"]).to(device)
    model_drop.load_state_dict(ckpt_drop["model_state"])

    print("Prédiction de la séquence complète (modèle sans dropout)...")
    pred_sequence = predict_sequence(model, image, device)

    print("Courbe de compression costoclaviculaire (CLAV--K1, distance min prédite)...")
    distances = costoclavicular_distance(pred_sequence, clav_label=1, k1_label=args.k1_class)
    valid = ~np.isnan(distances)
    print(f"  {valid.sum()}/{n_frames} frames avec CLAV et K1 tous deux prédits")

    print("Incertitude MC Dropout par frame (30 passes, K1)...")
    entropies = []
    for i in range(n_frames):
        image_t = torch.from_numpy(image[:, :, i]).unsqueeze(0).unsqueeze(0).float().to(device)
        mean_probs, _ = mc_dropout_predict(model_drop, image_t, n_samples=30)
        entropy = predictive_entropy(mean_probs)
        entropies.append(float(entropy.mean()))
    entropies = np.array(entropies)

    print("Seg-Grad-CAM par frame (K1) et cohérence temporelle...")
    cam_tool = SegGradCAM(model, model.dec1)
    cams = []
    for i in range(n_frames):
        image_t = torch.from_numpy(image[:, :, i]).unsqueeze(0).unsqueeze(0).float().to(device)
        cam = cam_tool(image_t, args.k1_class)
        cams.append(cam)
    n_missing = sum(c is None for c in cams)
    print(f"  K1 absent de la prédiction sur {n_missing}/{n_frames} frames (CAM non défini)")
    consistency = temporal_saliency_consistency(cams)

    tag = f"phase3_{args.sequence}_{args.patient.replace(' ', '_')}_{args.serie.replace(' ', '')}"
    out_path = OUT_DIR / f"{tag}_dashboard.png"
    plot_phase3_dashboard(instance_numbers, distances, entropies, consistency, out_path,
                           title_suffix=f" -- {args.patient}, {args.serie}")
    print(f"\nFigure sauvegardée : {out_path}")

    csv_path = OUT_DIR / f"{tag}_per_frame.csv"
    per_frame_df = pd.DataFrame({
        "instance_number": instance_numbers, "distance_clav_k1": distances, "entropy": entropies,
    })
    per_frame_df.to_csv(csv_path, index=False)
    print(f"Données par frame sauvegardées : {csv_path}")

    print("\n=== Résumé ===")
    print(f"Distance costoclaviculaire : min={np.nanmin(distances):.1f}px, "
          f"max={np.nanmax(distances):.1f}px, moyenne={np.nanmean(distances):.1f}px")
    print(f"Entropie moyenne (incertitude) : {entropies.mean():.4f} (min={entropies.min():.4f}, "
          f"max={entropies.max():.4f})")
    valid_consistency = consistency[~np.isnan(consistency)]
    if len(valid_consistency):
        print(f"Cohérence temporelle Seg-Grad-CAM (Spearman frame-à-frame) : "
              f"moyenne={valid_consistency.mean():.3f}, min={valid_consistency.min():.3f}")

    valid = ~np.isnan(distances)
    if valid.sum() > 2:
        rho, pval = spearmanr(distances[valid], entropies[valid])
        print(f"\nCorrélation incertitude / distance CLAV-K1 (Spearman, n={valid.sum()}) : "
              f"rho={rho:.4f}, p={pval:.2e}")
        print("  (rho < 0 : l'incertitude monte quand la distance baisse, i.e. en cas de compression)")


if __name__ == "__main__":
    main()
