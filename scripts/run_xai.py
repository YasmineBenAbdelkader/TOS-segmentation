"""Phase 2 — construit et valide Seg-Grad-CAM + occlusion sur le modèle FLASH fold 1
(meilleur checkpoint disponible, Dice 0,7796 -- voir rapport §Choix du modèle XAI).
MC Dropout n'est pas inclus ici (nécessite un ré-entraînement, voir même section).

Usage: python scripts/run_xai.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.dataset import TOSDataset
from src.metrics import STRUCTURE_NAMES
from src.model import UNet2D
from src.splitting import get_kfold_splits
from src.visualization import plot_occlusion, plot_sanity_check, plot_seggradcam
from src.xai import SegGradCAM, cascading_randomization_test, occlusion_sensitivity

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "DATA_processed"
CKPT = ROOT / "checkpoints" / "checkpoint_FLASH_fold1.pt"
OUT_DIR = ROOT / "report" / "figures"
NEEDS_REVIEW = ["2017-110_01-0224-V1MR"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")

    checkpoint = torch.load(CKPT, weights_only=False, map_location=device)
    model = UNet2D(in_channels=1, n_classes=7, base_ch=32).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Checkpoint chargé : fold {checkpoint['fold']}, séquence {checkpoint['sequence']}, "
          f"epoch {checkpoint['epoch']}")

    splits = get_kfold_splits(PROCESSED / "manifest.csv", k=3, always_train=NEEDS_REVIEW)
    val_patients = splits[1]["val"]  # fold 1, cohérent avec le checkpoint
    val_ds = TOSDataset(PROCESSED, val_patients, crop=False, sequence="FLASH")
    print(f"{len(val_ds)} frames de validation disponibles ({len(val_patients)} patients)")

    cam_tool = SegGradCAM(model, model.dec1)

    rng = random.Random(0)
    sample_idxs = rng.sample(range(len(val_ds)), 4)

    print("\n=== Seg-Grad-CAM : 4 échantillons x 6 structures ===")
    for i, idx in enumerate(sample_idxs):
        image, label = val_ds[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image_batch).argmax(dim=1).squeeze(0).cpu()
        img_np = image.squeeze(0).numpy()

        for class_idx in range(1, 7):
            structure = STRUCTURE_NAMES[class_idx]
            gt_mask = (label == class_idx).numpy()
            pred_mask = (pred == class_idx).numpy()
            if not gt_mask.any() and not pred_mask.any():
                continue  # structure absente de cet échantillon, rien à expliquer
            cam = cam_tool(image_batch, class_idx)
            if cam is None:
                continue
            out_path = OUT_DIR / f"xai_seggradcam_sample{i}_{structure}.png"
            plot_seggradcam(img_np, cam, gt_mask, pred_mask, structure, out_path)
        print(f"  échantillon {i} (idx={idx}) : figures sauvegardées")

    print("\n=== Occlusion : 2 échantillons x 2 structures (K1 = pire, VSC = meilleure) ===")
    occlusion_structures = {"K1": 5, "VSC": 3}
    for i in sample_idxs[:2]:
        image, label = val_ds[i]
        image_batch = image.unsqueeze(0).to(device)
        img_np = image.squeeze(0).numpy()
        for name, class_idx in occlusion_structures.items():
            gt_mask = (label == class_idx).numpy()
            heatmap = occlusion_sensitivity(model, image_batch, class_idx, patch_size=16, stride=8)
            if heatmap is None:
                continue
            out_path = OUT_DIR / f"xai_occlusion_sample{sample_idxs.index(i)}_{name}.png"
            plot_occlusion(img_np, heatmap, gt_mask, name, out_path)
        print(f"  échantillon idx={i} : occlusion terminée")

    print("\n=== Test de sanity check (randomisation en cascade, Adebayo et al.) ===")
    sanity_idx = sample_idxs[0]
    image, label = val_ds[sanity_idx]
    image_batch = image.unsqueeze(0).to(device)
    for name, class_idx in [("K1", 5), ("VSC", 3)]:
        correlations = cascading_randomization_test(
            model, lambda m: m.dec1, image_batch, class_idx, seed=0,
        )
        if correlations is None:
            print(f"  {name} : structure absente de cet échantillon, sanity check ignoré")
            continue
        out_path = OUT_DIR / f"xai_sanity_check_{name}.png"
        plot_sanity_check(correlations, name, out_path)
        print(f"  {name} : corrélations = {[round(c, 3) for c in correlations]}")

    print(f"\nToutes les figures sauvegardées dans {OUT_DIR}")


if __name__ == "__main__":
    main()
