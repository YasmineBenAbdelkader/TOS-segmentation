"""Phase 3 — analyse spécifique au caractère dynamique. Contrairement aux phases
précédentes (20 frames annotées par série), travaille sur la séquence dynamique
complète (autant de frames que présentes dans DATA/<patient>/<série>/, jusqu'à ~300),
sans vérité terrain -- toutes les métriques ici utilisent les prédictions du modèle
déjà entraîné et validé (Phase 1/2), jamais un label."""
from pathlib import Path

import numpy as np
import pydicom
import torch
from scipy.ndimage import distance_transform_edt

from scipy.ndimage import laplace

from src.preprocessing import n4_bias_correction, normalize_volume, resample_volume_and_mask


def load_full_sequence(series_dir, target_spacing=1.0):
    """Charge TOUTES les frames IM_*.dcm présentes dans series_dir (pas seulement les
    20 annotées -- fonctionne à l'identique avec 20 ou ~300 frames, aucun changement
    de code nécessaire une fois les frames manquantes ajoutées). Retourne (image,
    instance_numbers) : image (H, W, n_frames) prétraitée (N4 + rééchantillonnage +
    normalisation, même pipeline que Phase 0), instance_numbers la liste triée des
    InstanceNumber DICOM correspondant à l'axe frame."""
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("IM_*.dcm"),
                    key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber))
    instance_numbers = [int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber) for f in files]

    first = pydicom.dcmread(files[0])
    spacing = float(first.PixelSpacing[0])
    image = np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in files], axis=-1)

    image = n4_bias_correction(image)
    dummy_label = np.zeros_like(image, dtype=np.uint8)  # resample_volume_and_mask veut un label ; jeté ensuite
    image, _ = resample_volume_and_mask(image, dummy_label, current_spacing=spacing, target_spacing=target_spacing)
    image = normalize_volume(image)
    return image, instance_numbers


@torch.no_grad()
def predict_sequence(model, image, device, batch_size=8):
    """Prédiction (argmax) frame par frame sur la séquence complète. image : (H, W,
    n_frames) déjà prétraitée. Retourne (n_frames, H, W) uint8."""
    model.eval()
    n_frames = image.shape[-1]
    preds = []
    for start in range(0, n_frames, batch_size):
        batch = image[:, :, start:start + batch_size]
        batch_t = torch.from_numpy(batch).permute(2, 0, 1).unsqueeze(1).float().to(device)  # (B,1,H,W)
        logits = model(batch_t)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.uint8)


def costoclavicular_distance(pred_sequence, clav_label=1, k1_label=5):
    """Distance minimale entre les masques CLAV et K1 prédits, par frame -- approxime
    l'espace costoclaviculaire réel (distance entre les deux structures osseuses qui
    le bornent), plus pertinent anatomiquement qu'une distance centroïde-à-centroïde
    pour évaluer une compression. NaN si l'une des deux structures est absente de la
    prédiction sur cette frame (pas de distance définie)."""
    n_frames = pred_sequence.shape[0]
    distances = np.full(n_frames, np.nan)
    for i in range(n_frames):
        clav_mask = pred_sequence[i] == clav_label
        k1_mask = pred_sequence[i] == k1_label
        if not clav_mask.any() or not k1_mask.any():
            continue
        dt = distance_transform_edt(~clav_mask)
        distances[i] = dt[k1_mask].min()
    return distances


def image_sharpness(image_2d, bbox=None, margin=15):
    """Variance du Laplacien -- mesure classique de netteté (plus la variance est
    élevée, plus l'image contient de hautes fréquences nettes ; le flou/mouvement
    lisse ces hautes fréquences et fait chuter la variance). bbox=(y0,y1,x0,x1) :
    si fourni, calculée uniquement dans cette région (+marge), pour tester si c'est
    la qualité locale autour de la structure qui compte, pas la netteté globale de
    l'image (dominée par le fond/autres tissus)."""
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        h, w = image_2d.shape
        y0, y1 = max(0, y0 - margin), min(h, y1 + margin)
        x0, x1 = max(0, x0 - margin), min(w, x1 + margin)
        image_2d = image_2d[y0:y1, x0:x1]
    if image_2d.size == 0:
        return float("nan")
    return float(laplace(image_2d).var())


def structure_bbox(pred_mask, labels):
    """Boîte englobante (y0,y1,x0,x1) couvrant toutes les classes de `labels`
    présentes dans pred_mask -- utilisée pour délimiter la région locale de
    image_sharpness autour de CLAV+K1. None si aucune des classes n'est présente."""
    mask = np.isin(pred_mask, labels)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def temporal_saliency_consistency(cam_sequence):
    """Corrélation de rang de Spearman entre cartes de saillance de frames
    consécutives -- une anatomie réelle bouge en douceur, donc une explication fidèle
    au mouvement réel devrait rester fortement corrélée frame à frame (proche de 1) ;
    une chute erratique (proche de 0, ou qui oscille) suggère que la saillance ne
    suit pas un phénomène anatomique continu. cam_sequence : liste de cartes (H, W)
    ou None (structure absente de cette frame -- corrélation NaN pour cette paire).
    Retourne un tableau de longueur n_frames-1 (une valeur par paire consécutive)."""
    from scipy.stats import spearmanr
    correlations = []
    for i in range(len(cam_sequence) - 1):
        a, b = cam_sequence[i], cam_sequence[i + 1]
        if a is None or b is None:
            correlations.append(float("nan"))
            continue
        rho, _ = spearmanr(a.flatten(), b.flatten())
        correlations.append(rho)
    return np.array(correlations)
