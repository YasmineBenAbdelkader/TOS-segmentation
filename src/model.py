import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """2x (conv 3x3 -> BatchNorm -> ReLU) — le bloc de base standard, rien d'exotique
    (voir choix méthodologique : privilégier les briques bien comprises pour ne pas
    confondre un artefact de méthodologie avec un vrai phénomène d'interprétabilité)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """U-Net 2D simple (Ronneberger et al. 2015), volontairement pas état de l'art —
    baseline de référence pour la Phase 1, pas l'objectif de performance du projet
    (voir chapitre Contexte : l'interprétabilité est la contribution centrale, pas la
    métrique de segmentation).

    4 niveaux de descente/remontée, base_ch=32 par défaut (léger, cohérent avec un
    dataset de 20 frames annotées par série).

    dropout : si >0, ajoute un Dropout2d après bottleneck/enc4/dec4 uniquement (pas
    à chaque couche) -- suit Kendall et al. (2015, "Bayesian SegNet"), qui trouvent
    que le dropout après chaque couche est un régularisateur trop fort et ralentit
    l'apprentissage ; ils le placent sur les couches centrales de
    l'encodeur-décodeur. Taux 0,2 (plus conservateur que le 0,5 de Bayesian SegNet)
    choisi pour ce dataset déjà petit (Phase 2, XAI -- voir rapport, MC Dropout).
    dropout=0 (défaut) préserve l'architecture exacte utilisée pour les 6 checkpoints
    du baseline Phase 1, chargeables sans changement (Dropout2d n'a pas de paramètres
    appris, ne modifie pas le state_dict)."""

    def __init__(self, in_channels=1, n_classes=7, base_ch=32, dropout=0.0):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = ConvBlock(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = ConvBlock(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = ConvBlock(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = ConvBlock(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], n_classes, kernel_size=1)

        self.drop_enc4 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_bottleneck = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_dec4 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.drop_enc4(self.enc4(self.pool(e3)))
        b = self.drop_bottleneck(self.bottleneck(self.pool(e4)))

        d4 = self.drop_dec4(self.dec4(torch.cat([self.up4(b), e4], dim=1)))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # logits (N, n_classes, H, W) — softmax appliqué dans la loss
