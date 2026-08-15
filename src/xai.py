"""Phase 2 — boîte à outils XAI. Deux méthodes complémentaires (gradient vs
perturbation) implémentées d'abord, sans ré-entraînement (voir justification dans le
rapport, chapitre Méthodologie XAI) : Seg-Grad-CAM et sensibilité par occlusion.
Un test de randomisation (Adebayo et al. 2018) permet de vérifier qu'une carte de
saillance reflète réellement les poids appris avant de lui faire confiance."""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


class SegGradCAM:
    """Seg-Grad-CAM (Vinogradova, Dibrov & Lopez, 2020, arXiv:2002.11434) --
    adaptation de Grad-CAM (Selvaraju et al. 2017) à la segmentation : au lieu de
    rétropropager depuis un score de classe scalaire (classification), on
    rétropropage depuis la somme des logits d'une classe sur une région d'intérêt
    (par défaut la région prédite pour cette structure), ce qui donne une carte de
    saillance spatiale cohérente avec une sortie dense plutôt qu'un seul vecteur de
    classe."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image, class_idx, region_mask=None):
        """image : (1,1,H,W). class_idx : index de structure cible (1..6, fond=0
        exclu par convention). region_mask : (H,W) bool -- région sur laquelle
        sommer les logits avant rétropropagation (défaut : région prédite,
        argmax==class_idx ; passer le masque de vérité terrain permet de comparer
        "où le modèle regarde pour prédire juste" à "où il regarde sur la vraie
        étendue de la structure").
        Renvoie None si la classe est absente de la région (rien à expliquer)."""
        self.model.zero_grad()
        logits = self.model(image)

        if region_mask is None:
            pred = logits.argmax(dim=1).squeeze(0)
            region_mask = (pred == class_idx)
        if region_mask.sum() == 0:
            return None

        score = logits[0, class_idx][region_mask].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP par canal (Grad-CAM original)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


@torch.no_grad()
def occlusion_sensitivity(model, image, class_idx, region_mask=None, patch_size=16, stride=8,
                           baseline_value=0.0):
    """Sensibilité par occlusion (Zeiler & Fergus 2014) -- couvre systématiquement
    l'image par patchs avec une valeur neutre (fond du z-score, proche de 0), mesure
    la chute du score de la classe cible sur la région d'intérêt à chaque occlusion.
    Méthode par perturbation, indépendante du gradient -- une région identifiée par
    les deux méthodes (Seg-Grad-CAM et occlusion) est un signal plus robuste qu'une
    seule des deux. Coûteux ((H/stride)*(W/stride) passes avant) -- réservé à
    quelques échantillons qualitatifs, pas à l'ensemble de validation."""
    model.eval()
    H, W = image.shape[-2:]

    base_logits = model(image)
    if region_mask is None:
        pred = base_logits.argmax(dim=1).squeeze(0)
        region_mask = (pred == class_idx)
    if region_mask.sum() == 0:
        return None
    base_score = base_logits[0, class_idx][region_mask].sum().item()

    heatmap = torch.zeros(H, W)
    counts = torch.zeros(H, W)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.clone()
            occluded[..., y:y + patch_size, x:x + patch_size] = baseline_value
            logits = model(occluded)
            score = logits[0, class_idx][region_mask].sum().item()
            heatmap[y:y + patch_size, x:x + patch_size] += (base_score - score)
            counts[y:y + patch_size, x:x + patch_size] += 1
    counts[counts == 0] = 1
    return (heatmap / counts).numpy()


def _ordered_layers_top_to_bottom(model):
    """Ordre des couples (nom, module) à poids, de la sortie vers l'entrée --
    utilisé par le test de randomisation en cascade. Suit l'ordre inverse du
    forward() de UNet2D (src/model.py), pas l'ordre de déclaration dans __init__."""
    names = [
        "out_conv", "dec1", "up1", "dec2", "up2", "dec3", "up3", "dec4", "up4",
        "bottleneck", "enc4", "enc3", "enc2", "enc1",
    ]
    modules = dict(model.named_modules())
    ordered = []
    for name in names:
        block = modules[name]
        for sub_name, sub_module in block.named_modules():
            if isinstance(sub_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d)):
                ordered.append(sub_module)
    return ordered


def cascading_randomization_test(model, target_layer_getter, image, class_idx, region_mask=None, seed=0):
    """Test de randomisation en cascade (Adebayo et al. 2018, "Sanity Checks for
    Saliency Maps") -- randomise progressivement les poids du modèle depuis la
    sortie vers l'entrée (réinitialisation, pas juste bruitage), recalcule la carte
    de saillance à chaque étape, et mesure sa corrélation de rang de Spearman avec
    la carte originale (modèle entraîné intact). Une vraie explication doit se
    dégrader nettement dès les premières couches randomisées (corrélation qui chute
    vers 0) ; une carte qui reste fortement corrélée malgré la randomisation ne
    reflète pas les poids appris -- Adebayo et al. montrent que Guided Backprop et
    Guided Grad-CAM échouent ce test, ce qui motive de le faire ici avant de faire
    confiance à Seg-Grad-CAM sur ce projet.

    target_layer_getter : fonction (model) -> module, pour ré-instancier SegGradCAM
    après chaque copie du modèle (les hooks doivent être ré-attachés à la copie).
    Renvoie la liste des corrélations de Spearman, une par étape de randomisation
    (0 = modèle intact, corrélation = 1.0 par construction)."""
    torch.manual_seed(seed)
    model_copy = copy.deepcopy(model)
    cam_tool = SegGradCAM(model_copy, target_layer_getter(model_copy))
    original_cam = cam_tool(image, class_idx, region_mask=region_mask)
    if original_cam is None:
        return None
    original_flat = original_cam.flatten()

    correlations = [1.0]
    layers = _ordered_layers_top_to_bottom(model_copy)
    for layer in layers:
        layer.reset_parameters()
        cam = cam_tool(image, class_idx, region_mask=region_mask)
        if cam is None:
            correlations.append(float("nan"))
            continue
        rho, _ = spearmanr(original_flat, cam.flatten())
        correlations.append(rho)
    return correlations
