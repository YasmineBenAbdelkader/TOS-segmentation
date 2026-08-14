from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ROI fixe (y0, y1, x0, x1), calcule sur les 51 series a geometrie resolue :
# vrai min/max de l'etendue des structures sur ces 51 series (pas un percentile --
# garanti couvrir toutes les series par construction, pas juste verifie apres coup) +
# marge de 10px, arrondi a des dimensions divisibles par 16 (compatible avec les 4
# niveaux de pooling du U-Net). Verifie : 0/51 series ne debordent. Approche "coarse"
# allegee (statistiques de population plutot qu'un modele de localisation entraine)
# -- voir justification report/sections/04_phase1_baseline.tex.
#
# Limite assumee : calcule uniquement sur les 51 series a geometrie resolue -- les 5
# series encore exclues (0222 x4, 0223/FLASH D) ne sont pas couvertes par cette
# analyse. Si elles sont reintegrees apres verification manuelle, revalider que leurs
# structures tombent bien dans ce crop avant de les utiliser telles quelles.
ROI_CROP = (96, 256, 23, 263)  # -> 160 x 240 px, contre 320 x 320 d'origine


class TOSDataset(Dataset):
    """Dataset 2D pour le U-Net de Phase 1 : chaque item est une frame individuelle
    (pas un volume entier) — cohérent avec une architecture 2D qui traite chaque image
    indépendamment. Charge depuis DATA_processed/ (déjà aligné, N4, rééchantillonné,
    normalisé par scripts/build_dataset.py).

    image : (1, H, W) float32, déjà normalisée (clip percentile + z-score).
    label : (H, W) long, valeurs 0 (fond) à 6 (structures canoniques, voir CANON).
    Recadrage ROI_CROP appliqué avant l'augmentation si crop=True (défaut) — jamais de
    recadrage différent entre train et val, c'est une constante fixe, pas apprise.
    """

    def __init__(self, processed_dir, patients, transform=None, crop=True):
        self.processed_dir = Path(processed_dir)
        manifest = pd.read_csv(self.processed_dir / "manifest.csv")
        ok = manifest[(manifest.status == "OK") & (manifest.patient.isin(patients))]
        if ok.empty:
            raise ValueError(f"Aucune série 'OK' pour les patients fournis : {patients}")

        self.index = []  # (patient, serie, frame_idx)
        for _, r in ok.iterrows():
            label_path = self.processed_dir / r.patient / r.serie / "label.npy"
            n_frames = np.load(label_path, mmap_mode="r").shape[-1]
            self.index.extend((r.patient, r.serie, f) for f in range(n_frames))

        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        patient, serie, frame_idx = self.index[idx]
        series_dir = self.processed_dir / patient / serie

        image = np.load(series_dir / "image.npy", mmap_mode="r")[:, :, frame_idx]
        label = np.load(series_dir / "label.npy", mmap_mode="r")[:, :, frame_idx]

        if self.crop:
            y0, y1, x0, x1 = ROI_CROP
            image = image[y0:y1, x0:x1]
            label = label[y0:y1, x0:x1]

        image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0)
        label = torch.from_numpy(np.array(label, dtype=np.int64))

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label
