"""Augmentation de données jointe image/masque (Phase 1) — voir
report/sections/04_phase1_baseline.tex pour la justification complète.

Paramètres volontairement plus conservateurs que les défauts nnU-Net (rotation
±30°, scaling 0.7-1.4) : nos structures sont petites et déjà bien cadrées après la
Phase 0, une augmentation agressive ajouterait du bruit plutôt qu'une variabilité
anatomique réaliste. Jamais de flip horizontal (question D/G = vraies images miroir
anatomiques, non tranchée avec l'encadrant).
"""
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class JointAugmentation:
    """Chaque transformation a sa propre probabilité d'application indépendante
    (convention nnU-Net) — pas un choix exclusif entre transformations. Rotation et
    scaling appliqués identiquement à l'image (interpolation bilinéaire) et au
    masque (plus-proche-voisin, jamais d'interpolation continue sur des labels
    entiers — même principe que le rééchantillonnage de la Phase 0)."""

    def __init__(
        self,
        p_affine=0.5, rotation_deg=12, scale_range=(0.9, 1.1),
        p_elastic=0.3, elastic_alpha=15.0, elastic_sigma=4.0,
        p_intensity=0.5, gamma_range=(0.85, 1.15), brightness_range=(0.9, 1.1),
    ):
        self.p_affine = p_affine
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.p_intensity = p_intensity
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range

    def __call__(self, image, label):
        # image: (1, H, W) float ; label: (H, W) long -> (1, H, W) le temps du traitement
        label = label.unsqueeze(0).float()

        if random.random() < self.p_affine:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            scale = random.uniform(*self.scale_range)
            image = TF.affine(image, angle=angle, translate=[0, 0], scale=scale, shear=0,
                               interpolation=TF.InterpolationMode.BILINEAR)
            label = TF.affine(label, angle=angle, translate=[0, 0], scale=scale, shear=0,
                               interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < self.p_elastic:
            h, w = image.shape[-2:]
            displacement = T.ElasticTransform.get_params(
                alpha=[self.elastic_alpha, self.elastic_alpha],
                sigma=[self.elastic_sigma, self.elastic_sigma],
                size=[h, w],
            )
            image = TF.elastic_transform(image, displacement, interpolation=TF.InterpolationMode.BILINEAR)
            label = TF.elastic_transform(label, displacement, interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < self.p_intensity:
            # gamma correction suppose des valeurs dans [0,1] -- l'image est z-scoree
            # (peut etre negative), on remappe temporairement avant/apres
            gamma = random.uniform(*self.gamma_range)
            img_min, img_max = image.min(), image.max()
            img01 = ((image - img_min) / (img_max - img_min + 1e-8)).clamp(0, 1) ** gamma
            image = img01 * (img_max - img_min) + img_min

        if random.random() < self.p_intensity:
            brightness = random.uniform(*self.brightness_range)
            image = image * brightness

        return image, label.squeeze(0).long()
