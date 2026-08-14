import numpy as np
import SimpleITK as sitk


def normalize_volume(volume, lower_pct=0.5, upper_pct=99.5):
    """Clip aux percentiles [lower_pct, upper_pct] puis z-score, calculé sur ce
    volume seul (jamais de stats globales partagées entre séquences/patients — l'IRM
    n'a pas d'échelle d'intensité universelle, contrairement au CT).

    Le z-score par cas est le défaut nnU-Net pour l'IRM (Isensee et al. 2021). Le clip
    percentile en amont N'EST PAS ce défaut (chez nnU-Net le clip percentile est
    spécifique au schéma CT, avec des percentiles globaux sur tout le dataset) — c'est
    un ajout motivé empiriquement ici par un point hyperintense de flux vasculaire trouvé
    sur ASC en FLASH (EDA), qui sinon gonfle l'écart-type et écrase le contraste du
    reste de l'image. Vérifié empiriquement (scripts/validate_normalization.py) que le
    clip ne détruit pas ce contraste.
    """
    lo, hi = np.percentile(volume, [lower_pct, upper_pct])
    clipped = np.clip(volume, lo, hi)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-8)


def n4_bias_correction(volume, shrink_factor=4):
    """Corrige l'inhomogénéité du champ magnétique (N4ITK) sur un volume (H, W, n_slices).

    On sous-échantillonne (shrink_factor) pour estimer le champ de biais — coûteux
    à pleine résolution — puis on applique ce champ à l'image originale en pleine
    résolution, plutôt que de corriger l'image sous-échantillonnée elle-même.

    Ici les "slices" sont en réalité des frames temporelles d'une même coupe anatomique
    (IRM dynamique) : le champ de biais (propriété de l'antenne/position patient) varie
    peu dans le temps, donc l'estimer sur la pile entière comme un pseudo-volume 3D est
    une approximation raisonnable, pas un abus de la méthode.
    """
    image = sitk.GetImageFromArray(np.transpose(volume, (2, 0, 1)).astype(np.float32))
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    shrunk_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    shrunk_mask = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(shrunk_image, shrunk_mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)

    corrected = image / sitk.Exp(log_bias_field)
    corrected_arr = sitk.GetArrayFromImage(corrected)
    return np.transpose(corrected_arr, (1, 2, 0)).astype(np.float32)


def resample_volume_and_mask(image, label, current_spacing, target_spacing=1.0):
    """Rééchantillonne (H, W, n_frames) de current_spacing (mm, isotrope en plan) vers
    target_spacing. Interpolation linéaire pour l'image, plus-proche-voisin pour le
    masque (jamais d'interpolation continue sur des labels entiers, ça créerait des
    valeurs de classe inexistantes aux frontières). Le nombre de frames (axe temporel,
    pas spatial) n'est jamais rééchantillonné.
    """
    if np.isclose(current_spacing, target_spacing):
        return image, label

    h, w, n_frames = image.shape
    scale = current_spacing / target_spacing
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    def _resample(volume_hwt, interpolator, dtype):
        sitk_img = sitk.GetImageFromArray(np.transpose(volume_hwt, (2, 0, 1)).astype(np.float64))
        sitk_img.SetSpacing((current_spacing, current_spacing, 1.0))
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing((target_spacing, target_spacing, 1.0))
        resampler.SetSize((new_w, new_h, n_frames))
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        out = resampler.Execute(sitk_img)
        return np.transpose(sitk.GetArrayFromImage(out), (1, 2, 0)).astype(dtype)

    image_r = _resample(image, sitk.sitkLinear, np.float32)
    label_r = _resample(label, sitk.sitkNearestNeighbor, np.uint8)
    return image_r, label_r
