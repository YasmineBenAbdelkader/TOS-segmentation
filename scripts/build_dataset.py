"""Phase 0 — construit le dataset consolidé (image, masque) pour les 56 séries.

Pipeline par série : load_aligned_pair (DICOM brut) -> N4 -> resampling vers 1.0mm
si besoin -> normalisation (clip percentile + z-score). Sauvegarde chaque série dans
DATA_processed/<patient>/<series>/{image.npy,label.npy} et écrit un manifeste global.

Les 5 séries à géométrie non résolue sont exclues (status != "OK" de load_aligned_pair).
2017-110_01-0224-V1MR est traitée normalement mais marquée `a_verifier_manuellement=True`
dans le manifeste (outlier de barycentre récurrent trouvé en EDA, voir
report/sections/02_exploration.tex section "Outliers de barycentre") — à confirmer sous
3D Slicer avant de l'utiliser sans réserve pour l'entraînement.

Usage: python scripts/build_dataset.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pydicom

from src.data_loading import load_aligned_pair
from src.preprocessing import normalize_volume, n4_bias_correction, resample_volume_and_mask

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DATA"
OUT = ROOT / "DATA_processed"
SERIES_NAMES = ["FLASH D", "FLASH G", "RSSFP D", "RSSFP G"]
TARGET_SPACING = 1.0
NEEDS_REVIEW = {"2017-110_01-0224-V1MR"}  # outlier de barycentre recurrent, voir EDA


def main():
    OUT.mkdir(exist_ok=True)
    rows = []

    for patient_dir in sorted(DATA.iterdir()):
        if not patient_dir.is_dir():
            continue
        for series_name in SERIES_NAMES:
            series_dir = patient_dir / series_name
            seg_path = series_dir / "Segmentation_PB.seg.nrrd"
            if not seg_path.exists():
                continue

            t0 = time.time()
            image, label, status = load_aligned_pair(series_dir, seg_path)

            row = {
                "patient": patient_dir.name, "serie": series_name,
                "status": status, "a_verifier_manuellement": patient_dir.name in NEEDS_REVIEW,
                "spacing_original_mm": None, "resampled": False, "shape_finale": None,
                "temps_s": None,
            }

            if status != "OK":
                print(f"EXCLU (geometrie non resolue) : {patient_dir.name} / {series_name}")
                rows.append(row)
                continue

            first_dcm = sorted(series_dir.glob("IM_*.dcm"))[0]
            spacing = float(pydicom.dcmread(first_dcm, stop_before_pixels=True).PixelSpacing[0])
            row["spacing_original_mm"] = spacing

            image = n4_bias_correction(image)
            if not np.isclose(spacing, TARGET_SPACING):
                image, label = resample_volume_and_mask(image, label, spacing, TARGET_SPACING)
                row["resampled"] = True
            image = normalize_volume(image)

            out_dir = OUT / patient_dir.name / series_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "image.npy", image.astype(np.float32))
            np.save(out_dir / "label.npy", label.astype(np.uint8))

            row["shape_finale"] = str(image.shape)
            row["temps_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"OK ({row['temps_s']}s) : {patient_dir.name} / {series_name} -> {image.shape}"
                  + ("  [A VERIFIER MANUELLEMENT]" if row["a_verifier_manuellement"] else ""))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "manifest.csv", index=False)

    n_ok = (manifest.status == "OK").sum()
    print(f"\n{n_ok} / {len(manifest)} séries traitées avec succès.")
    print(f"Manifeste : {OUT / 'manifest.csv'}")
    if manifest.a_verifier_manuellement.any():
        flagged = manifest[manifest.a_verifier_manuellement & (manifest.status == "OK")]
        print(f"\n{len(flagged)} série(s) traitée(s) mais A VERIFIER MANUELLEMENT avant usage sans reserve :")
        print(flagged[["patient", "serie"]].to_string(index=False))


if __name__ == "__main__":
    main()
