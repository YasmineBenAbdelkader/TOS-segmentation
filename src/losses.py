import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(dataset, n_classes=7, max_samples=None):
    """Poids de la Cross-Entropy = fréquence inverse des pixels par classe, mesurée
    empiriquement sur le train set (pas une estimation a priori). Le déséquilibre
    dominant est fond vs structures (le fond occupe l'essentiel des 320x320 pixels) ;
    le ratio de taille entre structures (4,3x, voir EDA) est secondaire par rapport à
    ça."""
    counts = torch.zeros(n_classes, dtype=torch.float64)
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(n):
        _, label = dataset[i]
        counts += torch.bincount(label.flatten(), minlength=n_classes).double()
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.sum() * n_classes  # normalisé autour de 1 en moyenne
    return weights.float()


class DiceLoss(nn.Module):
    """Dice multi-classe (softmax), moyennée sur les classes — chaque structure pèse
    autant que les autres dans la loss peu importe sa taille, contrairement à la CE
    seule qui serait dominée par le fond."""

    def __init__(self, n_classes=7, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    """Dice + Cross-Entropy pondérée — choix méthodologique standard (pas exotique),
    voir report/sections/03_phase0_prep.tex pour les sources."""

    def __init__(self, n_classes=7, class_weights=None):
        super().__init__()
        self.dice = DiceLoss(n_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        return self.dice(logits, target) + self.ce(logits, target)
