"""Valide empiriquement que le pipeline de Phase 0 (N4 puis clip percentile + z-score,
voir src/preprocessing.py) ne détruit pas le contraste structure/fond, en particulier
le contraste FLASH vs RSSFP établi dans l'EDA (notebooks/analyse_exploratoire_TOS.ipynb,
section 9) — ne pas se contenter d'un raisonnement théorique, mesurer.

Usage: python scripts/validate_normalization.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.data_loading import CANON, load_aligned_pair
from src.preprocessing import normalize_volume, n4_bias_correction

DATA = Path(__file__).resolve().parent.parent / "DATA"
OUT_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"

PATIENT = "2017-110 01-0229-V1MR"
SERIES = ("FLASH D", "RSSFP D")


def structure_contrast(image, label):
    """CNR par structure — métrique de contrôle qualité, pas une étape du pipeline."""
    bg = image[label == 0]
    out = {}
    for name, val in CANON.items():
        m = label == val
        if m.sum() == 0:
            continue
        out[name] = float((image[m].mean() - bg.mean()) / (bg.std() + 1e-8))
    return out


def main():
    data = {}
    for s in SERIES:
        series_dir = DATA / PATIENT / s
        seg_path = series_dir / "Segmentation_PB.seg.nrrd"
        image, label, status = load_aligned_pair(series_dir, seg_path)
        assert status == "OK", status
        n4 = n4_bias_correction(image)
        normed = normalize_volume(n4)
        data[s] = (image, label, n4, normed)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for row, s in enumerate(SERIES):
        image, _, n4, normed = data[s]
        axes[row, 0].hist(image.ravel(), bins=100, color="steelblue")
        axes[row, 0].set_title(f"{s} — brut")
        axes[row, 1].hist(n4.ravel(), bins=100, color="seagreen")
        axes[row, 1].set_title(f"{s} — après N4")
        axes[row, 2].hist(normed.ravel(), bins=100, color="darkorange")
        axes[row, 2].set_title(f"{s} — après N4 + clip[0.5,99.5] + z-score")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "validation_normalisation_hist.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Histogrammes sauvegardés : {fig_path}")

    print("\n=== CNR : brut vs après N4 vs après N4 + normalisation ===")
    rows = []
    for s in SERIES:
        image, label, n4, normed = data[s]
        cnr_raw = structure_contrast(image, label)
        cnr_n4 = structure_contrast(n4, label)
        cnr_norm = structure_contrast(normed, label)
        print(f"\n{s}:")
        for name in CANON:
            if name in cnr_raw:
                print(f"  {name:5s}  brut={cnr_raw[name]:+.3f}   apres_N4={cnr_n4[name]:+.3f}   "
                      f"normalise={cnr_norm[name]:+.3f}")
                rows.append({"serie": s, "structure": name, "cnr_brut": cnr_raw[name],
                             "cnr_n4": cnr_n4[name], "cnr_normalise": cnr_norm[name]})

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR.parent / "figures" / "validation_normalisation_cnr.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTable CNR sauvegardée : {out_csv}")


if __name__ == "__main__":
    main()
