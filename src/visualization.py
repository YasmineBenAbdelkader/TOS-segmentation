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


def generate_all_figures(history, final_metrics, model, val_dataset, device, out_dir, fold):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_history_csv(history, out_dir / f"phase1_fold{fold}_history.csv")
    plot_loss_curve(history, out_dir / f"phase1_fold{fold}_loss.png")
    plot_dice_curve(history, out_dir / f"phase1_fold{fold}_dice_curve.png")
    plot_train_val_gap(history, out_dir / f"phase1_fold{fold}_train_val_gap.png")
    plot_final_metrics_bar(
        final_metrics["dice"].tolist(), final_metrics["precision"].tolist(),
        final_metrics["recall"].tolist(), final_metrics["hausdorff95"].tolist(),
        out_dir / f"phase1_fold{fold}_final_metrics.png",
    )
    plot_qualitative(model, val_dataset, device, out_dir / f"phase1_fold{fold}_qualitative.png")
    print(f"Figures sauvegardées dans {out_dir} (préfixe phase1_fold{fold}_*)")
