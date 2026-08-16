"""Génération de courbes et figures de résultats pour la Phase 1 — inspiré des
figures standards des papiers de segmentation médicale (courbes d'entraînement,
diagramme en barres par structure, résultats qualitatifs image/vérité/prédiction)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_loading import CANON
from src.metrics import STRUCTURE_NAMES


def save_history_csv(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def plot_loss_curve(history, out_path, title="Perte d'entraînement (Dice + CE pondérée)"):
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["epoch"], df["train_loss"], marker="o", ms=3, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_dice_curve(history, out_path, n_classes=7, title="Dice de validation par structure"):
    """Une courbe par structure (hors fond, peu informatif) -- permet de repérer si
    une structure stagne pendant que les autres progressent (piste diagnostique citée
    dans la discussion sur le choix de loss : surveiller ASC en particulier)."""
    df = pd.DataFrame(history)
    dice_cols = np.array(df["val_dice_per_class"].tolist())
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for c in range(1, n_classes):
        ax.plot(df["epoch"], dice_cols[:, c], marker="o", ms=2, label=STRUCTURE_NAMES[c], color=cmap((c - 1) / 10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_train_val_gap(history, out_path, title="Dice train vs val (diagnostic de sur-apprentissage)"):
    """Dice train (sous-échantillon fixe, non augmenté) vs Dice val, moyennés sur les
    6 structures -- un écart qui se creuse est le signal direct de sur-apprentissage,
    plus lisible qu'une comparaison loss/Dice (métriques différentes). Pertinent pour
    juger l'effet de l'augmentation de données (voir rapport, discussion Phase 1) :
    l'augmentation devrait réduire cet écart, pas seulement changer le Dice val seul."""
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    if "train_dice_per_class" in df.columns:
        mean_train_dice = np.array(df["train_dice_per_class"].tolist())[:, 1:].mean(axis=1)
        ax.plot(df["epoch"], mean_train_dice, color="steelblue", marker="o", ms=3, label="Dice train")
    mean_val_dice = np.array(df["val_dice_per_class"].tolist())[:, 1:].mean(axis=1)
    ax.plot(df["epoch"], mean_val_dice, color="darkorange", marker="s", ms=3, label="Dice val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice (moyen structures)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_final_metrics_bar(dice, precision, recall, hausdorff, out_path):
    """Diagramme en barres groupées -- format standard pour résumer les métriques
    finales par structure dans un papier (voir ex. tables/figures BraTS)."""
    names = STRUCTURE_NAMES[1:]
    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width, dice[1:], width, label="Dice", color="steelblue")
    axes[0].bar(x, precision[1:], width, label="Précision", color="seagreen")
    axes[0].bar(x + width, recall[1:], width, label="Rappel", color="darkorange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title("Dice / Précision / Rappel par structure")
    axes[0].grid(alpha=0.3, axis="y")

    hd = np.nan_to_num(np.array(hausdorff[1:]), nan=0.0)
    axes[1].bar(x, hd, color="firebrick")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_title("Hausdorff-95 par structure (pixels)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison_bar(dice_a, dice_b, hausdorff_a, hausdorff_b, label_a, label_b, out_path,
                         title_suffix=""):
    """Compare deux runs (ex. avec/sans augmentation, ou plus tard FLASH vs RSSFP vs
    pooled) sur les mêmes structures -- barres groupées Dice + Hausdorff-95
    côte à côte, réutilisable pour toute comparaison à 2 conditions."""
    names = STRUCTURE_NAMES[1:]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - width / 2, dice_a[1:], width, label=label_a, color="steelblue")
    axes[0].bar(x + width / 2, dice_b[1:], width, label=label_b, color="darkorange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title(f"Dice par structure{title_suffix}")
    axes[0].grid(alpha=0.3, axis="y")

    hd_a = np.nan_to_num(np.array(hausdorff_a[1:]), nan=0.0)
    hd_b = np.nan_to_num(np.array(hausdorff_b[1:]), nan=0.0)
    axes[1].bar(x - width / 2, hd_a, width, label=label_a, color="steelblue")
    axes[1].bar(x + width / 2, hd_b, width, label=label_b, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].legend()
    axes[1].set_title(f"Hausdorff-95 (pixels){title_suffix}")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_qualitative(model, dataset, device, out_path, n_samples=4, seed=0):
    """Grille image / vérité terrain / prédiction sur des échantillons de validation
    tirés au hasard -- figure quasi systématique dans les papiers de segmentation
    ('résultats qualitatifs'), complémentaire aux métriques agrégées."""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    cmap = plt.get_cmap("tab10")

    model.eval()
    fig, axes = plt.subplots(len(idxs), 3, figsize=(10, 3.3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(idxs):
        image, label = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        img_np = image.squeeze(0).numpy()
        label_np = label.numpy()

        for col, sub_title in enumerate(["Image", "Vérité terrain", "Prédiction"]):
            axes[row, col].imshow(img_np, cmap="gray")
            axes[row, col].set_title(sub_title if row == 0 else "")
            axes[row, col].axis("off")

        for name, val in CANON.items():
            color = cmap((val - 1) / 10)
            m_gt = label_np == val
            m_pred = pred == val
            if m_gt.sum():
                axes[row, 1].contour(m_gt, levels=[0.5], colors=[color], linewidths=1.5)
            if m_pred.sum():
                axes[row, 2].contour(m_pred, levels=[0.5], colors=[color], linewidths=1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_seggradcam(image, cam, gt_mask, pred_mask, structure_name, out_path):
    """Superposition image / carte Seg-Grad-CAM (jet, semi-transparente) / contours
    vérité terrain (vert) et prédiction (blanc pointillé) -- permet de juger d'un
    coup d'œil si la zone de saillance coïncide avec la structure réelle ou prédite."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    if gt_mask is not None and gt_mask.any():
        ax.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=1.5)
    if pred_mask is not None and pred_mask.any():
        ax.contour(pred_mask, levels=[0.5], colors=["white"], linewidths=1, linestyles="dashed")
    ax.set_title(f"Seg-Grad-CAM — {structure_name}\n(vert : vérité terrain, blanc : prédiction)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_occlusion(image, heatmap, gt_mask, structure_name, out_path):
    """Carte de sensibilité par occlusion -- rouge = zone dont l'occlusion fait le
    plus chuter le score de la structure, donc la plus "importante" pour le modèle
    au sens de la perturbation (complémentaire au gradient de Seg-Grad-CAM)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    vmax = max(abs(heatmap.min()), abs(heatmap.max()), 1e-8)
    im = ax.imshow(heatmap, cmap="coolwarm", alpha=0.6, vmin=-vmax, vmax=vmax)
    if gt_mask is not None and gt_mask.any():
        ax.contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=1.5)
    ax.set_title(f"Sensibilité par occlusion — {structure_name}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Chute du score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sanity_check(correlations, structure_name, out_path):
    """Courbe corrélation de Spearman vs étape de randomisation en cascade (Adebayo
    et al. 2018) -- une chute rapide vers 0 est le signe attendu d'une explication
    fidèle au modèle ; une courbe qui reste haute signale un outil à ne pas
    utiliser tel quel (voir chapitre Méthodologie XAI)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(correlations)), correlations, marker="o", color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Étape de randomisation en cascade (0 = modèle intact)")
    ax.set_ylabel("Corrélation de Spearman avec la carte originale")
    ax.set_ylim(-0.15, 1.05)
    ax.set_title(f"Test de randomisation (Adebayo et al. 2018) — {structure_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_uncertainty(image, entropy_map, gt_mask, pred_mask, error_mask, out_path):
    """Carte d'incertitude (entropie prédictive MC Dropout) + carte d'erreur
    juxtaposées -- permet de juger visuellement si l'incertitude est élevée
    précisément là où le modèle se trompe (signal de calibration utile) ou
    diffuse/décorrélée des erreurs (signal moins exploitable)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap="gray")
    im = axes[0].imshow(entropy_map, cmap="magma", alpha=0.6)
    if gt_mask is not None and gt_mask.any():
        axes[0].contour(gt_mask, levels=[0.5], colors=["lime"], linewidths=1.2)
    if pred_mask is not None and pred_mask.any():
        axes[0].contour(pred_mask, levels=[0.5], colors=["cyan"], linewidths=1, linestyles="dashed")
    axes[0].set_title("Incertitude (entropie MC Dropout)")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(error_mask, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Erreur de prédiction (rouge = pixel mal classé)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_phase3_dashboard(instance_numbers, distances, entropies, consistency, out_path,
                           title_suffix=""):
    """Tableau de bord Phase 3 -- 3 panneaux empilés partageant l'axe temporel (frame) :
    distance costoclaviculaire (compression), incertitude MC Dropout moyenne, et
    cohérence temporelle de la saillance (Seg-Grad-CAM). Empilés plutôt que
    superposés pour garder des échelles indépendantes lisibles (mm, entropie,
    corrélation de Spearman n'ont rien de commun)."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(instance_numbers, distances, marker="o", ms=3, color="firebrick")
    axes[0].set_ylabel("Distance CLAV–K1 (px ≈ mm)")
    axes[0].set_title(f"Courbe de compression costoclaviculaire{title_suffix}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(instance_numbers, entropies, marker="o", ms=3, color="steelblue")
    axes[1].set_ylabel("Entropie moyenne\n(incertitude MC Dropout)")
    axes[1].grid(alpha=0.3)

    mid_x = [(instance_numbers[i] + instance_numbers[i + 1]) / 2 for i in range(len(consistency))]
    axes[2].plot(mid_x, consistency, marker="o", ms=3, color="seagreen")
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Cohérence temporelle\n(Spearman, frames consécutives)")
    axes[2].set_xlabel("Frame (InstanceNumber)")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_all_figures(history, final_metrics, model, val_dataset, device, out_dir, tag):
    """tag : préfixe complet des fichiers générés (ex. 'baseline_FLASH_fold0'), pas
    seulement un numéro de fold — permet de distinguer les runs par séquence/modèle
    sans coupler cette fonction à la convention de nommage d'un script particulier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_history_csv(history, out_dir / f"{tag}_history.csv")
    plot_loss_curve(history, out_dir / f"{tag}_loss.png")
    plot_dice_curve(history, out_dir / f"{tag}_dice_curve.png")
    plot_train_val_gap(history, out_dir / f"{tag}_train_val_gap.png")
    plot_final_metrics_bar(
        final_metrics["dice"].tolist(), final_metrics["precision"].tolist(),
        final_metrics["recall"].tolist(), final_metrics["hausdorff95"].tolist(),
        out_dir / f"{tag}_final_metrics.png",
    )
    plot_qualitative(model, val_dataset, device, out_dir / f"{tag}_qualitative.png")
    print(f"Figures sauvegardées dans {out_dir} (préfixe {tag}_*)")
