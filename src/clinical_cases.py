"""Phase 4 — sélection et génération des cas cliniques curés pour la session de
validation avec l'encadrant radiologue. Réutilise les outils déjà validés
(Seg-Grad-CAM, MC Dropout, Phase 2/3) sur des frames choisies pour répondre à une
question précise déjà posée dans la thèse, pas au hasard."""
from pathlib import Path

import torch

from src.dataset import TOSDataset
from src.metrics import STRUCTURE_NAMES, dice_per_class
from src.xai import SegGradCAM, mc_dropout_predict, predictive_entropy


@torch.no_grad()
def per_frame_dice(model, dataset, class_idx, device):
    """Dice de la classe class_idx, frame par frame, sur tout le dataset (pas de
    MC Dropout ici -- juste une passe avant, peu coûteux). Renvoie une liste de
    (index, dice)."""
    model.eval()
    results = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        logits = model(image.unsqueeze(0).to(device))
        d = dice_per_class(logits, label.unsqueeze(0).to(device), n_classes=7)
        results.append((i, float(d[class_idx])))
    return results


def find_extreme_frame(model, dataset, class_idx, device, mode="worst"):
    """Index de la frame avec le Dice le plus bas ('worst') ou le plus haut
    ('best') pour class_idx, parmi les frames où la structure est présente dans la
    vérité terrain (Dice=0 par absence totale n'est pas une vraie erreur de
    segmentation, juste une frame sans intérêt pour ce cas)."""
    scores = per_frame_dice(model, dataset, class_idx, device)
    valid = [(i, d) for i, d in scores if dataset[i][1].eq(class_idx).any()]
    if not valid:
        return None
    valid.sort(key=lambda x: x[1])
    return valid[0][0] if mode == "worst" else valid[-1][0]


def build_case_card_data(model, model_dropout, dataset, frame_idx, target_class, device):
    """Calcule tout ce qu'il faut pour une carte de cas clinique sur une frame avec
    vérité terrain disponible : image, GT, prédiction, Seg-Grad-CAM (target_class),
    carte d'incertitude MC Dropout, Dice de la frame. Renvoie un dict."""
    image, label = dataset[frame_idx]
    image_batch = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_batch)
        pred = logits.argmax(dim=1).squeeze(0).cpu()
        dice = dice_per_class(logits, label.unsqueeze(0).to(device), n_classes=7)

    cam_tool = SegGradCAM(model, model.dec1)
    cam = cam_tool(image_batch, target_class)

    mean_probs, _ = mc_dropout_predict(model_dropout, image_batch, n_samples=30)
    entropy = predictive_entropy(mean_probs)

    return {
        "image": image.squeeze(0).numpy(), "label": label.numpy(), "pred": pred.numpy(),
        "cam": cam, "entropy": entropy,
        "dice_target": float(dice[target_class]),
        "structure_name": STRUCTURE_NAMES[target_class],
    }
