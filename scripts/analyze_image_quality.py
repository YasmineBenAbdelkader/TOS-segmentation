"""Phase 3 -- teste l'hypothèse alternative pour l'incertitude au cours du temps :
est-ce la qualité d'image locale (netteté), pas la distance CLAV--K1 brute, qui
prédit l'incertitude du modèle ? Réutilise les entropies déjà calculées et
sauvegardées (report/figures/phase3_*_per_frame.csv) -- ne relance pas MC Dropout
(coûteux), seulement le rechargement d'image (N4/resample/normalize) + une
prédiction simple (pas de dropout) pour délimiter la région locale.

Usage: python scripts/analyze_image_quality.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.model import UNet2D
from src.phase3 import image_sharpness, load_full_sequence, predict_sequence, structure_bbox

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "big_dataset"
OUT_DIR = ROOT / "report" / "figures"

# (patient, serie, sequence, fold) -- les 13 séries déjà traitées en Phase 3
RUNS = [
    ("2017-110 01-0230-V1MR", "FLASH D", "FLASH", 1),
    ("2017-110 01-0230-V1MR", "FLASH G", "FLASH", 1),
    ("2017-110 01-0231-V1MR", "FLASH D", "FLASH", 1),
    ("2017-110 01-0240-V1MR", "FLASH D", "FLASH", 1),
    ("2017-110 01-0228-V1MR", "FLASH D", "FLASH", 1),
    ("2017-110 01-0229-V1MR", "RSSFP D", "RSSFP", 0),
    ("2017-110 01-0229-V1MR", "RSSFP G", "RSSFP", 0),
    ("2017-110 01-0223-V1MR", "RSSFP D", "RSSFP", 0),
    ("2017-110 01-0223-V1MR", "RSSFP G", "RSSFP", 0),
    ("2017-110 01-0226-V1MR", "RSSFP D", "RSSFP", 0),
    ("2017-110 01-0226-V1MR", "RSSFP G", "RSSFP", 0),
    ("2017-110 01-0227-V1MR", "RSSFP D", "RSSFP", 0),
    ("2017-110 01-0227-V1MR", "RSSFP G", "RSSFP", 0),
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")
    model_cache = {}
    results = []

    for patient, serie, sequence, fold in RUNS:
        tag = f"phase3_{sequence}_{patient.replace(' ', '_')}_{serie.replace(' ', '')}"
        csv_path = OUT_DIR / f"{tag}_per_frame.csv"
        if not csv_path.exists():
            print(f"  [ignoré] {csv_path.name} introuvable")
            continue
        per_frame = pd.read_csv(csv_path)

        key = (sequence, fold)
        if key not in model_cache:
            ckpt = torch.load(ROOT / "checkpoints" / f"checkpoint_{sequence}_fold{fold}.pt",
                               weights_only=False, map_location=device)
            m = UNet2D(in_channels=1, n_classes=7, base_ch=32).to(device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            model_cache[key] = m
        model = model_cache[key]

        print(f"Chargement {patient}/{serie}...")
        image, instance_numbers = load_full_sequence(DATA_ROOT / patient / serie)
        pred_sequence = predict_sequence(model, image, device)

        global_sharp, local_sharp = [], []
        for i in range(image.shape[-1]):
            frame = image[:, :, i]
            global_sharp.append(image_sharpness(frame))
            bbox = structure_bbox(pred_sequence[i], labels=[1, 5])  # CLAV=1, K1=5
            local_sharp.append(image_sharpness(frame, bbox=bbox) if bbox else float("nan"))

        per_frame["sharpness_global"] = global_sharp
        per_frame["sharpness_local"] = local_sharp
        per_frame.to_csv(csv_path, index=False)  # complète le csv existant avec les 2 nouvelles colonnes

        valid = per_frame.dropna(subset=["entropy", "sharpness_local"])
        rho_local, p_local = spearmanr(valid["entropy"], valid["sharpness_local"])
        rho_global, p_global = spearmanr(per_frame["entropy"], per_frame["sharpness_global"])
        rho_dist_sharp, _ = spearmanr(valid["distance_clav_k1"], valid["sharpness_local"])

        results.append({
            "patient": patient, "serie": serie, "sequence": sequence,
            "rho_entropy_sharpness_local": rho_local, "p_local": p_local,
            "rho_entropy_sharpness_global": rho_global, "p_global": p_global,
            "rho_distance_sharpness_local": rho_dist_sharp,
        })
        print(f"  entropie vs netteté locale : rho={rho_local:.4f} (p={p_local:.2e})  "
              f"| netteté globale : rho={rho_global:.4f} (p={p_global:.2e})  "
              f"| distance vs netteté : rho={rho_dist_sharp:.4f}")

    results_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "phase3_image_quality_summary.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n=== Synthèse sur", len(results_df), "séries ===")
    for col in ["rho_entropy_sharpness_local", "rho_entropy_sharpness_global", "rho_distance_sharpness_local"]:
        print(f"{col} : mean={results_df[col].mean():.4f}  std={results_df[col].std():.4f}  "
              f"n_positif={int((results_df[col] > 0).sum())}/{len(results_df)}")
    print(f"\nRésultats sauvegardés dans {out_csv}")


if __name__ == "__main__":
    main()
