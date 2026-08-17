"""Phase 4 — génère les cartes de cas cliniques curés pour la session de
validation avec l'encadrant radiologue. 10 cas sur les 8 patients couverts par un
modèle avec dropout (FLASH fold 1, RSSFP fold 0 -- seuls folds où MC Dropout est
disponible, voir Phase 2), chacun choisi pour répondre à une question précise déjà
posée dans le rapport technique, pas au hasard.

Usage: python scripts/prepare_phase4_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.clinical_cases import build_case_card_data, find_extreme_frame
from src.dataset import TOSDataset
from src.data_loading import CANON
from src.model import UNet2D
from src.splitting import get_kfold_splits
from src.visualization import plot_clinical_case

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "DATA_processed"
OUT_DIR = ROOT / "report" / "figures" / "phase4_cases"
NEEDS_REVIEW = ["2017-110_01-0224-V1MR"]

# (patient, sequence, fold, structure, mode, justification courte)
CASES = [
    ("2017-110 01-0240-V1MR", "FLASH", 1, "K1", "worst",
     "Meilleur patient FLASH en moyenne (Dice global 0,82) mais K1 reste faible (0,57) -- "
     "la difficulté de K1 persiste-t-elle même sur un cas globalement facile ?"),
    ("2017-110 01-0240-V1MR", "FLASH", 1, "VSC", "best",
     "Même patient, sa meilleure structure (VSC, Dice 0,92) -- à quoi ressemble un cas "
     "où le modèle est confiant et juste ?"),
    ("2017-110_01-0226-V1MR", "RSSFP", 0, "K1", "worst",
     "Pire cas K1 sur les 8 patients couverts (Dice 0,395) -- le pire cas du chapitre."),
    ("2017-110_01-0226-V1MR", "RSSFP", 0, "MSC", "worst",
     "Même patient, 2e structure la plus faible (MSC, 0,68) -- ce patient est-il "
     "globalement difficile, ou seulement pour K1 ?"),
    ("2017-110 01-0229-V1MR", "RSSFP", 0, "MSC", "worst",
     "Patient dont la corrélation incertitude/compression n'était même pas significative "
     "en Phase 3 -- sa structure la plus faible (MSC, 0,50)."),
    ("2017-110 01-0230-V1MR", "FLASH", 1, "K1", "worst",
     "Patient pilote de toute l'analyse Phase 3 (courbe de compression, cohérence "
     "temporelle) -- son pire cas K1."),
    ("2017-110 01-0231-V1MR", "FLASH", 1, "K1", "worst",
     "Patient signalé en Phase 3 pour une distance CLAV-K1 anormalement basse (3,0px) "
     "``à vérifier visuellement'' -- jamais fait, à faire ici."),
    ("2017-110_01-0223-V1MR", "RSSFP", 0, "K1", "best",
     "Meilleur cas K1 côté RSSFP (Dice 0,669) -- contraste avec les pires cas ci-dessus."),
    ("2017-110_01-0228-V1MR", "FLASH", 1, "K1", "best",
     "Meilleur cas K1 côté FLASH (Dice 0,676) -- contraste avec les pires cas ci-dessus."),
    ("2017-110_01-0227-V1MR", "RSSFP", 0, "K1", "worst",
     "2e pire cas K1 (Dice 0,456) -- confirme-t-il le même type d'erreur que 0226 ?"),
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_cache = {}
    summary = []

    for i, (patient, sequence, fold, structure, mode, justification) in enumerate(CASES):
        class_idx = CANON[structure]
        key = (sequence, fold)
        if key not in model_cache:
            ckpt = torch.load(ROOT / "checkpoints" / f"checkpoint_{sequence}_fold{fold}.pt",
                               weights_only=False, map_location=device)
            m = UNet2D(in_channels=1, n_classes=7, base_ch=32).to(device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()

            ckpt_d = torch.load(ROOT / "checkpoints" / f"checkpoint_{sequence}_fold{fold}_dropout.pt",
                                 weights_only=False, map_location=device)
            m_d = UNet2D(in_channels=1, n_classes=7, base_ch=32, dropout=ckpt_d["dropout"]).to(device)
            m_d.load_state_dict(ckpt_d["model_state"])
            model_cache[key] = (m, m_d)
        model, model_dropout = model_cache[key]

        splits = get_kfold_splits(PROCESSED / "manifest.csv", k=3, always_train=NEEDS_REVIEW)
        val_patients = splits[fold]["val"]
        assert patient in val_patients, (
            f"{patient} n'est pas dans le fold {fold} de validation {sequence} : {val_patients}")
        dataset = TOSDataset(PROCESSED, [patient], crop=False, sequence=sequence)  # ce patient seul,
        # pas tout le fold -- sinon find_extreme_frame cherche la pire/meilleure frame sur les 4
        # patients mélangés et ignore silencieusement lequel est demandé (bug détecté et corrigé)

        print(f"[{i+1}/{len(CASES)}] {patient} / {sequence} / {structure} ({mode})...")
        frame_idx = find_extreme_frame(model, dataset, class_idx, device, mode=mode)
        if frame_idx is None:
            print(f"  [ignoré] structure {structure} absente de la vérité terrain pour ce patient")
            continue

        case_data = build_case_card_data(model, model_dropout, dataset, frame_idx, class_idx, device)
        tag = f"case{i+1:02d}_{patient.replace(' ', '_')}_{sequence}_{structure}_{mode}"
        out_path = OUT_DIR / f"{tag}.png"
        title = f"Cas {i+1} -- {patient} ({sequence})"
        plot_clinical_case(case_data, title, out_path, target_class=class_idx)

        summary.append({
            "cas": i + 1, "patient": patient, "sequence": sequence, "structure": structure,
            "mode": mode, "dice": case_data["dice_target"], "entropie_moyenne": float(case_data["entropy"].mean()),
            "justification": justification, "figure": out_path.name,
        })
        print(f"  Dice={case_data['dice_target']:.3f}  entropie moy.={case_data['entropy'].mean():.4f}  "
              f"-> {out_path.name}")

    summary_df = pd.DataFrame(summary)
    summary_csv = OUT_DIR / "cases_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n{len(summary_df)} cas générés. Résumé : {summary_csv}")


if __name__ == "__main__":
    main()
