import pandas as pd
from sklearn.model_selection import KFold


def get_kfold_splits(manifest_path, k=3, always_train=None, seed=42):
    """K folds au niveau patient (13 patients éligibles, N trop petit pour un split
    fixe train/val/test robuste — voir report/sections/03_phase0_prep.tex). Chaque
    patient est son propre groupe, présent une seule fois dans la liste à découper —
    KFold simple suffit, pas besoin de GroupKFold puisqu'il n'y a pas de répétition de
    groupe dans cette liste.

    k=3 par défaut (pas 5) : compromis coût de calcul / fiabilité pour la comparaison
    FLASH vs RSSFP à 2 modèles — voir discussion méthodologique. La comparaison finale
    ne repose de toute façon pas sur une moyenne par fold mais sur un appariement par
    patient (chaque patient passe exactement une fois en validation par modèle, quel
    que soit k), donc réduire k ne réduit pas le nombre de paires du test statistique
    final, seulement le coût d'entraînement.

    `always_train` : patients qui ne doivent jamais tomber en validation (ex.
    2017-110_01-0224-V1MR, non vérifié manuellement — si son annotation s'avère
    fautive, on ne veut pas que ça fausse une métrique de validation).
    """
    manifest = pd.read_csv(manifest_path)
    ok = manifest[manifest.status == "OK"]
    eligible = sorted(ok.patient.unique())
    always_train = set(always_train or [])
    foldable = [p for p in eligible if p not in always_train]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(foldable):
        train_patients = sorted([foldable[i] for i in train_idx] + list(always_train))
        val_patients = sorted([foldable[i] for i in val_idx])
        splits.append({"train": train_patients, "val": val_patients})
    return splits
