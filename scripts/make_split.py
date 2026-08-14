"""Phase 0 — split train/val/test au niveau patient, à partir du manifeste produit
par build_dataset.py. Seuls les patients ayant au moins une série "OK" sont éligibles
(2017-110_01-0222-V1MR n'a aucune série a geometrie resolue et est donc absent).

Avec seulement 13 patients éligibles, un split fixe train/val/test laisse très peu de
patients en val/test (l'évaluation sera bruitée) — une validation croisée (GroupKFold
au niveau patient) serait plus robuste pour la Phase 1. Le split fixe ci-dessous est un
point de départ simple, pas une recommandation définitive : à rediscuter avant Phase 1.

Usage: python scripts/make_split.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "DATA_processed" / "manifest.csv"
OUT = ROOT / "DATA_processed" / "split.json"

N_VAL = 2
N_TEST = 2
SEED = 42


def main():
    manifest = pd.read_csv(MANIFEST)
    ok = manifest[manifest.status == "OK"]
    eligible_patients = sorted(ok.patient.unique())
    print(f"{len(eligible_patients)} patients éligibles (>= 1 série OK) sur {manifest.patient.nunique()} au total.")

    excluded_patients = sorted(set(manifest.patient.unique()) - set(eligible_patients))
    if excluded_patients:
        print(f"Patients entièrement exclus (0 série OK) : {excluded_patients}")

    flagged = sorted(ok[ok.a_verifier_manuellement].patient.unique())
    if flagged:
        print(f"Patients avec des séries a verifier manuellement (forcés en train, jamais val/test tant "
              f"que non confirmés) : {flagged}")

    # les patients a verifier manuellement ne doivent jamais tomber en val/test : si leurs
    # annotations sont erronees, ca fausserait les metriques d'evaluation. On les force en
    # train (impact dilue) et on ne pioche val/test que parmi les patients confirmes.
    splittable = [p for p in eligible_patients if p not in flagged]
    rng = __import__("random").Random(SEED)
    shuffled = splittable.copy()
    rng.shuffle(shuffled)

    test_patients = shuffled[:N_TEST]
    val_patients = shuffled[N_TEST:N_TEST + N_VAL]
    train_patients = shuffled[N_TEST + N_VAL:] + flagged

    split = {
        "seed": SEED,
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
        "excluded_no_usable_series": excluded_patients,
        "included_but_needs_manual_review": flagged,
    }

    OUT.write_text(json.dumps(split, indent=2, ensure_ascii=False))
    print(f"\nTrain: {len(train_patients)} patients — {train_patients}")
    print(f"Val:   {len(val_patients)} patients — {val_patients}")
    print(f"Test:  {len(test_patients)} patients — {test_patients}")
    print(f"\nSplit sauvegardé : {OUT}")
    print("\nATTENTION : avec si peu de patients en val/test, considérer une validation "
          "croisée (GroupKFold patient) plutôt qu'un split fixe pour la Phase 1.")


if __name__ == "__main__":
    main()
