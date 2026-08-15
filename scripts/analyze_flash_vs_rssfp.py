"""Comparaison statistique FLASH vs RSSFP au niveau patient (pas au niveau fold --
voir report/rapport_baseline_flash_rssfp.tex, §Validation croisée pour la
justification : n=3 folds ne permet pas d'atteindre p<0.05 avec un test de Wilcoxon,
n=12 patients appariés le permet).

Usage: python scripts/analyze_flash_vs_rssfp.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "report" / "figures" / "baseline_per_patient_metrics.csv"


def holm_bonferroni(pvals, alpha=0.05):
    """Correction de Holm-Bonferroni (moins conservatrice que Bonferroni simple,
    contrôle le même FWER) -- pas de dépendance à statsmodels (non installé),
    implémentation directe : trier p croissant, comparer p_(i) à alpha/(m-i+1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    significant = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[i] <= threshold:
            significant[i] = True
        else:
            break  # Holm : dès qu'un test échoue, tous les suivants (p plus grand) échouent aussi
    return significant


def main():
    df = pd.read_csv(CSV)
    df = df[df.structure != "fond"]

    structures = sorted(df.structure.unique())
    results = []
    for structure in structures:
        sub = df[df.structure == structure]
        flash = sub[sub.sequence == "FLASH"].set_index("patient")["dice"]
        rssfp = sub[sub.sequence == "RSSFP"].set_index("patient")["dice"]
        common_patients = sorted(set(flash.index) & set(rssfp.index))
        f = flash.loc[common_patients].values
        r = rssfp.loc[common_patients].values

        diff = f - r
        stat, pval = wilcoxon(f, r)
        results.append({
            "structure": structure, "n_patients": len(common_patients),
            "flash_median": float(pd.Series(f).median()), "rssfp_median": float(pd.Series(r).median()),
            "median_diff_flash_minus_rssfp": float(pd.Series(diff).median()),
            "mean_diff_flash_minus_rssfp": float(diff.mean()),
            "n_flash_better": int((diff > 0).sum()), "n_rssfp_better": int((diff < 0).sum()),
            "n_tied": int((diff == 0).sum()),
            "wilcoxon_stat": float(stat), "p_raw": float(pval),
        })

    results_df = pd.DataFrame(results)
    results_df["significant_holm"] = holm_bonferroni(results_df["p_raw"].tolist())

    out_csv = ROOT / "report" / "figures" / "flash_vs_rssfp_stats.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"n patients appariés par structure : {results_df.n_patients.tolist()}")
    print()
    for _, r in results_df.iterrows():
        sig = "significatif (Holm)" if r.significant_holm else "non significatif"
        print(f"{r.structure:6s}  Dice médian FLASH={r.flash_median:.4f}  RSSFP={r.rssfp_median:.4f}  "
              f"diff(F-R) médiane={r.median_diff_flash_minus_rssfp:+.4f}  "
              f"FLASH meilleur sur {r.n_flash_better}/{r.n_patients} patients  "
              f"p_raw={r.p_raw:.4f}  {sig}")
    print(f"\nRésultats sauvegardés dans {out_csv}")


if __name__ == "__main__":
    main()
