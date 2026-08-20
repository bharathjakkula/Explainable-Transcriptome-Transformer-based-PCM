import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
INPUT_FILE = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca", "SUBTYPE_PCA_DEG_expression_matrix.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "04_prognosis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NON_GENE_COLS = ["PATIENT_ID", "OS_STATUS", "OS_MONTHS", "SUBTYPE", "AGE", "SEX", "TUMOR_STAGE"]
TOP_N_KM_PLOTS = 50


def run_univariate_cox(data, genes, outdir):
    results = []
    for gene in genes:
        try:
            t = data[["OS_MONTHS", "OS_STATUS", gene]].dropna()
            if t[gene].nunique() < 3:
                continue
            cph = CoxPHFitter()
            cph.fit(t, duration_col="OS_MONTHS", event_col="OS_STATUS")
            s = cph.summary.loc[gene]
            results.append({
                "Gene": gene, "HR": s["exp(coef)"],
                "HR_lower": s["exp(coef) lower 95%"], "HR_upper": s["exp(coef) upper 95%"],
                "p_value": s["p"], "-log10(p)": -np.log10(s["p"]),
            })
        except Exception:
            continue
    res = pd.DataFrame(results).sort_values("p_value")
    res.to_csv(os.path.join(outdir, "Prognostic_DEGs_Overall.csv"), index=False)
    return res


def plot_km(data, gene, outdir, hr_data=None):
    med = data[gene].median()
    hi = data[data[gene] >= med]
    lo = data[data[gene] < med]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 5))
    kmf.fit(hi["OS_MONTHS"], hi["OS_STATUS"], label=f"{gene} High")
    ax = kmf.plot(ci_show=False)
    kmf.fit(lo["OS_MONTHS"], lo["OS_STATUS"], label=f"{gene} Low")
    kmf.plot(ax=ax, ci_show=False)

    p = logrank_test(hi["OS_MONTHS"], lo["OS_MONTHS"],
                      event_observed_A=hi["OS_STATUS"], event_observed_B=lo["OS_STATUS"]).p_value

    hr = l = u = None
    if hr_data is not None:
        r = hr_data[hr_data["Gene"] == gene]
        if not r.empty:
            hr, l, u = r["HR"].iat[0], r["HR_lower"].iat[0], r["HR_upper"].iat[0]
    if hr is None:
        try:
            t = data[["OS_MONTHS", "OS_STATUS", gene]].dropna()
            c = CoxPHFitter()
            c.fit(t, duration_col="OS_MONTHS", event_col="OS_STATUS")
            s = c.summary.loc[gene]
            hr, l, u = s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]
        except Exception:
            hr = "N/A"

    txt = (f"Log-rank p = {p:.4f}\nHR = {hr:.3f} ({l:.3f} - {u:.3f})"
           if hr != "N/A" and l is not None else f"Log-rank p = {p:.4f}\nHR = {hr}")
    plt.annotate(txt, xy=(0.05, 0.05), xycoords="axes fraction",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
    plt.title(f"Kaplan-Meier Curve: {gene}")
    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"KM_{gene}.png"), dpi=300)
    plt.close()
    return {"gene": gene, "hr": None if hr == "N/A" else hr, "hr_ci_lower": l, "hr_ci_upper": u, "logrank_p": p}


def main():
    df = pd.read_csv(INPUT_FILE)
    required = {"PATIENT_ID", "OS_STATUS", "OS_MONTHS"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input file must contain {required}")

    genes = [c for c in df.columns if c not in NON_GENE_COLS and c != "Unnamed: 0"]
    print(f"Running univariate Cox regression on {len(genes)} signature genes...")
    res = run_univariate_cox(df, genes, OUTPUT_DIR)

    if not res.empty:
        km_results = []
        for gene in res.nsmallest(TOP_N_KM_PLOTS, "p_value")["Gene"]:
            km_results.append(plot_km(df, gene, OUTPUT_DIR, res))
        pd.DataFrame(km_results).to_csv(os.path.join(OUTPUT_DIR, "KM_Plot_Results.csv"), index=False)
        print(f"Top 5 prognostic genes:\n{res.head()}")

    print(f"\nDone. Prognostic analysis complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
