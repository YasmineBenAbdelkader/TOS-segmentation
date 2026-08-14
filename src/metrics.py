import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.data_loading import CANON

STRUCTURE_NAMES = ["fond"] + list(CANON.keys())  # index 0..6, aligné sur les labels canoniques


@torch.no_grad()
def dice_per_class(logits, target, n_classes=7, smooth=1e-5):
    """Dice par classe sur un batch — jamais un seul chiffre macro-moyenné (les
    structures varient trop en taille, voir EDA §distribution de taille) : os/muscle
    contre nerf/vaisseaux n'ont pas le même Dice attendu."""
    pred = logits.argmax(dim=1)
    dice = torch.zeros(n_classes)
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = pred_c.sum().float() + target_c.sum().float()
        dice[c] = (2 * intersection + smooth) / (union + smooth)
    return dice


@torch.no_grad()
def precision_recall_per_class(logits, target, n_classes=7, smooth=1e-5):
    """Précision/rappel par classe — le Dice mélange faux positifs et faux négatifs
    en un seul chiffre ; ça distingue sur- vs sous-segmentation, utile cliniquement
    (ex. sous-segmenter PB, un nerf, est plus préoccupant que le sur-segmenter)."""
    pred = logits.argmax(dim=1)
    precision = torch.zeros(n_classes)
    recall = torch.zeros(n_classes)
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        precision[c] = (tp + smooth) / (tp + fp + smooth)
        recall[c] = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def _surface(mask):
    """Voxels de bord (mask & ~erosion(mask)) — même définition que le score de
    gradient utilisé pour valider l'alignement en EDA (cohérence de convention)."""
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def _hausdorff95_2d(pred_mask, gt_mask):
    """Hausdorff-95 (pixels) entre deux masques binaires 2D, par distance de surface
    symétrique. Cas limites : les deux vides -> 0 (rien à comparer, pas d'erreur) ; un
    seul vide -> NaN (non défini, à exclure des moyennes plutôt que de biaiser avec une
    valeur arbitraire comme 0 ou une distance max)."""
    pred_empty, gt_empty = not pred_mask.any(), not gt_mask.any()
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float("nan")

    pred_surf, gt_surf = _surface(pred_mask), _surface(gt_mask)
    dt_gt = distance_transform_edt(~gt_mask)
    dt_pred = distance_transform_edt(~pred_mask)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def _nsd_2d(pred_mask, gt_mask, tau=2.0):
    """Normalized Surface Distance entre deux masques binaires 2D, tolérance tau en
    pixels -- fraction de la frontière prédite/vérité qui tombe à moins de tau px de
    l'autre frontière. Mêmes conventions que _hausdorff95_2d (surface = _surface,
    distance transform contre le masque plein) pour rester cohérent : deux masques
    vides -> 1.0 (accord parfait, rien à comparer) ; un seul vide -> NaN (non défini,
    exclu des moyennes plutôt que biaisé par un 0 arbitraire)."""
    pred_empty, gt_empty = not pred_mask.any(), not gt_mask.any()
    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return float("nan")

    pred_surf, gt_surf = _surface(pred_mask), _surface(gt_mask)
    dt_gt = distance_transform_edt(~gt_mask)
    dt_pred = distance_transform_edt(~pred_mask)

    close_pred = (dt_gt[pred_surf] <= tau).sum()
    close_gt = (dt_pred[gt_surf] <= tau).sum()
    return float((close_pred + close_gt) / (pred_surf.sum() + gt_surf.sum()))


@torch.no_grad()
def nsd_per_class(logits, target, n_classes=7, tau=2.0):
    """Normalized Surface Distance (surface Dice) par classe, tolérance tau=2px par
    défaut (spacing rééchantillonné à 1mm en Phase 0, donc ~2mm). Recommandée par
    Maier-Hein et al. 2024 ("Metrics reloaded", Nature Methods) en complément du Dice
    (recouvrement) et du Hausdorff-95 (frontière, sensible aux valeurs aberrantes) --
    ces métriques sont insuffisamment corrélées pour se substituer l'une à l'autre.
    Coûteux comme le Hausdorff-95 (distance transform par image) -- réservé à
    l'évaluation finale, pas à chaque epoch."""
    pred = logits.argmax(dim=1).cpu().numpy()
    target_np = target.cpu().numpy()
    batch_size = pred.shape[0]

    nsd = torch.zeros(n_classes)
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        for b in range(batch_size):
            v = _nsd_2d(pred[b] == c, target_np[b] == c, tau=tau)
            if not np.isnan(v):
                nsd[c] += v
                counts[c] += 1
    result = nsd / counts.clamp(min=1)
    result[counts == 0] = float("nan")
    return result


@torch.no_grad()
def hausdorff95_per_class(logits, target, n_classes=7):
    """Hausdorff-95 par classe, moyenné sur le batch (NaN ignorés — cas où une
    structure est absente d'une frame mais prédite, ou l'inverse). Plus coûteux que le
    Dice (distance transform par image), à utiliser en évaluation finale, pas à chaque
    epoch. Sensible aux erreurs de forme/contour que le Dice peut manquer (un petit
    faux positif isolé loin de la structure affecte à peine le Dice mais beaucoup le
    Hausdorff) — complémentaire au Dice, pas redondant, voir plan méthodologique
    Phase 1 du rapport."""
    pred = logits.argmax(dim=1).cpu().numpy()
    target_np = target.cpu().numpy()
    batch_size = pred.shape[0]

    hd = torch.zeros(n_classes)
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        for b in range(batch_size):
            d = _hausdorff95_2d(pred[b] == c, target_np[b] == c)
            if not np.isnan(d):
                hd[c] += d
                counts[c] += 1
    # NaN (pas 0) quand une classe n'a aucun echantillon valide dans le batch --
    # diviser par clamp(min=1) sans ca renverrait silencieusement un faux "score
    # parfait" de 0 plutot que "non defini", et fausserait la moyenne dans evaluate().
    result = hd / counts.clamp(min=1)
    result[counts == 0] = float("nan")
    return result
