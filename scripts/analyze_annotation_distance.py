"""Phase 3 -- dernière sous-analyse prévue : l'incertitude est-elle plus élevée sur
les frames éloignées de toute frame annotée à l'entraînement ? Réutilise les
entropies déjà calculées et sauvegardées (mêmes 13 séries, per_frame.csv) --
ajoute juste la distance à l'annotation la plus proche, calculée en retrouvant les
20 frames annotées dans la séquence complète par comparaison de pixels.

Usage: python scripts/analyze_annotation_distance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.phase3 import distance_to_nearest_annotated, match_annotated_frames

ROOT = Path(__file__).resolve().parent.parent
DATA_ANNOTATED = ROOT / "DATA"
DATA_FULL = ROOT / "big_dataset"
OUT_DIR = ROOT / "report" / "figures"


def resolve_patient_dir(root, patient):
    """DATA/ et big_dataset/ n'utilisent pas toujours la même convention
    espace/underscore pour le même patient (incohérence déjà connue dans ce
    projet, cf. manifest.csv) -- essaie les deux plutôt que d'échouer
    silencieusement sur un dossier introuvable."""
    candidates = [patient, patient.replace(" ", "_"), patient.replace("_", " ")]
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Aucun dossier trouvé pour {patient!r} sous {root} (essayé {candidates})")

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


def main():
    results = []
    for patient, serie, sequence, fold in RUNS:
        tag = f"phase3_{sequence}_{patient.replace(' ', '_')}_{serie.replace(' ', '')}"
        csv_path = OUT_DIR / f"{tag}_per_frame.csv"
        if not csv_path.exists():
            print(f"  [ignoré] {csv_path.name} introuvable")
            continue
        per_frame = pd.read_csv(csv_path)

        annotated_dir = resolve_patient_dir(DATA_ANNOTATED, patient) / serie
        full_dir = resolve_patient_dir(DATA_FULL, patient) / serie
        matched, unmatched = match_annotated_frames(annotated_dir, full_dir)
        print(f"{patient}/{serie} : {len(matched)}/20 frames annotées retrouvées dans la "
              f"séquence complète" + (f" ({len(unmatched)} non retrouvées)" if unmatched else ""))
        if not matched:
            print("  [ignoré] aucune correspondance -- impossible de calculer la distance")
            continue

        per_frame["distance_annotation"] = distance_to_nearest_annotated(
            per_frame["instance_number"].tolist(), matched)
        per_frame.to_csv(csv_path, index=False)

        rho, p = spearmanr(per_frame["entropy"], per_frame["distance_annotation"])
        results.append({
            "patient": patient, "serie": serie, "sequence": sequence,
            "n_annotated_matched": len(matched), "n_annotated_unmatched": len(unmatched),
            "rho_entropy_distance_annotation": rho, "p": p,
        })
        print(f"  entropie vs distance-à-l'annotation : rho={rho:.4f} (p={p:.2e})")

    results_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "phase3_annotation_distance_summary.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"\n=== Synthèse sur {len(results_df)} séries ===")
    col = "rho_entropy_distance_annotation"
    print(f"{col} : mean={results_df[col].mean():.4f}  std={results_df[col].std():.4f}  "
          f"n_positif={int((results_df[col] > 0).sum())}/{len(results_df)}")
    print(f"\nRésultats sauvegardés dans {out_csv}")


if __name__ == "__main__":
    main()
