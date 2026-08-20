import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
# NPASS_Pan_active_hits_cleaned_calculated.csv is produced OUTSIDE this codebase:
# the 292 pan-active hits (09b output) were docked (Discovery Studio) and run
# through ADMET prediction software (GUI tools, not scripted -- see repo README).
# Place that tool's exported CSV here before running this script.
INPUT_CSV = os.path.join(BASE_DIR, "input", "docking_admet", "NPASS_Pan_active_hits_cleaned_calculated.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "10_admet_sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE = dict(hbd=5, hba=10, rb=10, tpsa=140, mw=500, alogp=5, sol=2, bbb=2)
CORE_LEADS = ["NPC139056", "NPC196231", "NPC471997"]   # order preserved for reporting

CONTINUOUS_PARAMS = {
    "hbd":   ("Num_H_Donors",       "H-Bond Donors (<=5)"),
    "hba":   ("Num_H_Acceptors",    "H-Bond Acceptors (<=10)"),
    "rb":    ("Num_RotatableBonds", "Rotatable Bonds (<=10)"),
    "tpsa":  ("ADMET_PSA_2D",       "TPSA (<=140 A^2)"),
    "mw":    ("Molecular_Weight",   "Molecular Weight (<=500)"),
    "alogp": ("ALogP",              "ALogP (<=5)"),
}
ORDINAL_PARAMS = {
    "sol": ("ADMET_Solubility_Level", "ADMET Solubility Level (>=2)"),
    "bbb": ("ADMET_BBB_Level",        "ADMET BBB Level (>=2)"),
}
BOOLEAN_PARAMS = {
    "cyp2d6": ("ADMET_EXT_CYP2D6#Prediction",      "CYP2D6 inhibition filter"),
    "hepato": ("ADMET_EXT_Hepatotoxic#Prediction", "Hepatotoxicity filter"),
    "ppb":    ("ADMET_EXT_PPB#Prediction",         "Plasma-protein binding filter"),
}


def apply_filter(df, params, drop_bool=None):
    drop_bool = drop_bool or set()
    mask = pd.Series(True, index=df.index)
    for key, (col, _) in CONTINUOUS_PARAMS.items():
        mask &= df[col] <= params[key]
    for key, (col, _) in ORDINAL_PARAMS.items():
        mask &= df[col] >= params[key]
    for key, (col, _) in BOOLEAN_PARAMS.items():
        if key not in drop_bool:
            mask &= (df[col] == False)
    return df[mask]


def lead_status(hit_df):
    """Per-lead retained flags, in CORE_LEADS order."""
    names = set(hit_df["Name"])
    return {lead: (lead in names) for lead in CORE_LEADS}


def lead_status_str(status_dict):
    return "; ".join(f"{k}:{'RETAINED' if v else 'LOST'}" for k, v in status_dict.items())


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"[FATAL] Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} pan-active compounds.")

    # ---- Modification 20: hard-stop baseline validation ----
    baseline_hits = apply_filter(df, BASELINE)
    baseline_names = sorted(baseline_hits["Name"])
    expected = sorted(CORE_LEADS)
    print(f"\nBaseline hit count: {len(baseline_hits)}")
    print(f"Baseline hit names: {baseline_names}")
    if len(baseline_hits) != 3 or baseline_names != expected:
        print("\n[FATAL] Baseline filter did NOT reproduce the expected 3 leads "
              f"{expected}. Got {len(baseline_hits)} hits: {baseline_names}.")
        print("STOPPING EXECUTION -- fix the filter/column mapping before proceeding.")
        sys.exit(1)
    print("[OK] Baseline validated: exactly the 3 expected leads reproduced.\n")
    baseline_n = len(baseline_hits)

    detailed_rows = []
    summary_rows = []   # one row per filter -> consolidated manuscript table

    # ============================================================
    # 1. Univariate sweep: continuous parameters (5-point, full curve)
    # ============================================================
    print("=" * 78)
    print("ONE-AT-A-TIME SENSITIVITY: continuous Lipinski/ADMET cutoffs")
    print("=" * 78)
    continuous_ranked = []
    for key, (col, label) in CONTINUOUS_PARAMS.items():
        base_val = BASELINE[key]
        curve_hits = {}
        for pct in [-20, -10, 0, 10, 20]:
            val = base_val * (1 + pct / 100)
            params = dict(BASELINE)
            params[key] = val
            hits = apply_filter(df, params)
            n = len(hits)
            dHits = n - baseline_n
            pctChange = (dHits / baseline_n * 100) if baseline_n else np.nan
            status = lead_status(hits)
            curve_hits[pct] = n
            detailed_rows.append({
                "Filter": label, "Shift": f"{pct:+d}%" if pct else "0% (baseline)",
                "Threshold_value": round(val, 2), "Hit_count": n,
                "Delta_hits_vs_baseline": dHits, "Percent_change": round(pctChange, 1),
                "Lead_status": lead_status_str(status),
                **{f"{lead}_retained": v for lead, v in status.items()}
            })
        # sensitivity coefficient using the +/-20% extremes
        d_hits_range = curve_hits[20] - curve_hits[-20]
        d_threshold_range = base_val * 0.4  # from -20% to +20% = 40% of baseline value
        S = d_hits_range / d_threshold_range if d_threshold_range else np.nan
        continuous_ranked.append((label, curve_hits, S, max(curve_hits.values()) - min(curve_hits.values())))
        print(f"  {label:28s}  -20%={curve_hits[-20]:2d}  -10%={curve_hits[-10]:2d}  "
              f"base={curve_hits[0]:2d}  +10%={curve_hits[10]:2d}  +20%={curve_hits[20]:2d}"
              f"   |  S={S:.4f} hits/unit")
        summary_rows.append({
            "Filter": label, "Shift_definition": "+/-20% of published value",
            "Baseline_hits": curve_hits[0],
            "Relaxed_hits": curve_hits[20], "Strict_hits": curve_hits[-20],
            "dHits_relaxed": curve_hits[20] - curve_hits[0],
            "dHits_strict": curve_hits[-20] - curve_hits[0],
            "Sensitivity_coefficient_hits_per_unit": round(S, 4),
        })

    # ============================================================
    # 2. Univariate sweep: ordinal ADMET level parameters
    # ============================================================
    print("\n" + "=" * 78)
    print("ONE-AT-A-TIME SENSITIVITY: ordinal ADMET level cutoffs (+/-1 level)")
    print("=" * 78)
    ordinal_ranked = []
    for key, (col, label) in ORDINAL_PARAMS.items():
        base_val = BASELINE[key]
        curve_hits = {}
        for delta, tag in [(-1, "-1 level"), (0, "0 (baseline)"), (1, "+1 level")]:
            val = base_val + delta
            params = dict(BASELINE)
            params[key] = val
            hits = apply_filter(df, params)
            n = len(hits)
            dHits = n - baseline_n
            pctChange = (dHits / baseline_n * 100) if baseline_n else np.nan
            status = lead_status(hits)
            curve_hits[delta] = n
            detailed_rows.append({
                "Filter": label, "Shift": tag, "Threshold_value": val,
                "Hit_count": n, "Delta_hits_vs_baseline": dHits,
                "Percent_change": round(pctChange, 1), "Lead_status": lead_status_str(status),
                **{f"{lead}_retained": v for lead, v in status.items()}
            })
        rng = max(curve_hits.values()) - min(curve_hits.values())
        ordinal_ranked.append((label, curve_hits, rng))
        print(f"  {label:28s}  -1={curve_hits[-1]:2d}  base={curve_hits[0]:2d}  +1={curve_hits[1]:2d}")
        summary_rows.append({
            "Filter": label, "Shift_definition": "+/-1 discrete ADMET level",
            "Baseline_hits": curve_hits[0],
            "Relaxed_hits": curve_hits[1], "Strict_hits": curve_hits[-1],
            "dHits_relaxed": curve_hits[1] - curve_hits[0],
            "dHits_strict": curve_hits[-1] - curve_hits[0],
            "Sensitivity_coefficient_hits_per_unit": "n/a (ordinal, 1-level step)",
        })

    # ============================================================
    # 3. Individual contribution of ADMET liability filters
    # ============================================================
    print("\n" + "=" * 78)
    print("INDIVIDUAL CONTRIBUTION OF ADMET LIABILITY FILTERS")
    print("=" * 78)
    boolean_ranked = []
    for key, (col, label) in BOOLEAN_PARAMS.items():
        for state, tag in [(True, "Applied (baseline)"), (False, "Not applied")]:
            hits = apply_filter(df, BASELINE, drop_bool=(set() if state else {key}))
            n = len(hits)
            dHits = n - baseline_n
            pctChange = (dHits / baseline_n * 100) if baseline_n else np.nan
            status = lead_status(hits)
            detailed_rows.append({
                "Filter": label, "Shift": tag, "Threshold_value": "boolean",
                "Hit_count": n, "Delta_hits_vs_baseline": dHits,
                "Percent_change": round(pctChange, 1), "Lead_status": lead_status_str(status),
                **{f"{lead}_retained": v for lead, v in status.items()}
            })
            if not state:
                not_applied_n = n
        rng = not_applied_n - baseline_n
        boolean_ranked.append((label, rng))
        print(f"  {label:32s}  Applied={baseline_n:2d}  Not applied={not_applied_n:2d}  "
              f"(dHits={rng:+d})")
        summary_rows.append({
            "Filter": label, "Shift_definition": "Applied vs. Not applied (boolean)",
            "Baseline_hits": baseline_n,
            "Relaxed_hits": not_applied_n, "Strict_hits": baseline_n,
            "dHits_relaxed": not_applied_n - baseline_n, "dHits_strict": 0,
            "Sensitivity_coefficient_hits_per_unit": "n/a (boolean, on/off)",
        })

    # ============================================================
    # 4. Global stringency sweep (all continuous+ordinal shifted together)
    # ============================================================
    print("\n" + "=" * 78)
    print("GLOBAL STRINGENCY SWEEP (all continuous+ordinal cutoffs shifted together)")
    print("=" * 78)
    global_rows = []
    for pct in [-20, -10, 0, 10, 20]:
        params = dict(BASELINE)
        for key in CONTINUOUS_PARAMS:
            params[key] = BASELINE[key] * (1 + pct / 100)
        level_shift = round(pct / 20)
        for key in ORDINAL_PARAMS:
            params[key] = BASELINE[key] - level_shift
        hits = apply_filter(df, params)
        n = len(hits)
        dHits = n - baseline_n
        status = lead_status(hits)
        global_rows.append({
            "Global_shift": f"{pct:+d}%" if pct else "0% (baseline)",
            "Direction": "stricter" if pct < 0 else ("looser" if pct > 0 else "baseline"),
            "Hit_count": n, "Delta_hits_vs_baseline": dHits,
            "Percent_change": round(dHits / baseline_n * 100, 1) if baseline_n else np.nan,
            "Lead_status": lead_status_str(status),
        })
        print(f"  {pct:+4d}%  hits={n:2d}  dHits={dHits:+d}  {lead_status_str(status)}")
    global_df = pd.DataFrame(global_rows)

    # ============================================================
    # 5. Ranking: most / least sensitive filters
    # ============================================================
    print("\n" + "=" * 78)
    print("FILTER SENSITIVITY RANKING (by hit-count range across its own sweep)")
    print("=" * 78)
    ranking = []
    for label, curve, S, rng in continuous_ranked:
        ranking.append((label, rng))
    for label, curve, rng in ordinal_ranked:
        ranking.append((label, rng))
    for label, rng in boolean_ranked:
        ranking.append((label, rng))
    ranking.sort(key=lambda x: -x[1])
    for i, (label, rng) in enumerate(ranking, start=1):
        print(f"  Rank {i:2d}: {label:32s}  hit-count range = {rng}")
    most_sensitive = ranking[0]

    min_rng = ranking[-1][1]
    tied_at_min = [label for label, rng in ranking if rng == min_rng]
    if len(tied_at_min) > 1:
        least_sensitive_phrase = (
            f"{len(tied_at_min)} of the 11 criteria were tied for least influential, each "
            f"showing zero measured effect on hit count (hit-count range = {min_rng}): "
            f"{', '.join(tied_at_min)}"
        )
    else:
        least_sensitive_phrase = f"the {tied_at_min[0]} was uniquely least influential"
    least_sensitive = (tied_at_min[0], min_rng)  # kept for any numeric/legacy use

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path_csv = os.path.join(OUTPUT_DIR, "ST_ADMET_Sensitivity_detailed.csv")
    detailed_df.to_csv(detailed_path_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path_csv = os.path.join(OUTPUT_DIR, "ST_ADMET_Sensitivity_summary.csv")
    summary_df.to_csv(summary_path_csv, index=False)
    summary_path_xlsx = os.path.join(OUTPUT_DIR, "ST_ADMET_Sensitivity_summary.xlsx")
    summary_df.to_excel(summary_path_xlsx, index=False)

    global_path = os.path.join(OUTPUT_DIR, "ST_ADMET_Sensitivity_global.csv")
    global_df.to_csv(global_path, index=False)

    ranking_df = pd.DataFrame(ranking, columns=["Filter", "Hit_count_range"])
    ranking_path = os.path.join(OUTPUT_DIR, "ST_ADMET_Sensitivity_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)

    print(f"\nSaved:\n  {detailed_path_csv}\n  {summary_path_csv}\n  {summary_path_xlsx}"
          f"\n  {global_path}\n  {ranking_path}")

    # ============================================================
    # FIGURES: (A) kept 2-panel bar chart, (B) new sequential filter funnel chart
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    param_labels = [v[1] for v in CONTINUOUS_PARAMS.values()]
    shifts = [-20, -10, 0, 10, 20]
    width = 0.15
    x = np.arange(len(param_labels))
    for i, pct in enumerate(shifts):
        counts = [continuous_ranked[j][1][pct] for j in range(len(param_labels))]
        ax1.bar(x + (i - 2) * width, counts, width, label=f"{pct:+d}%" if pct else "baseline")
    ax1.axhline(baseline_n, color="red", linestyle="--", lw=1, alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Number of hit compounds")
    ax1.set_title("(A) Lipinski/ADMET continuous cutoffs\n(hit count fully insensitive, +/-20%)")
    ax1.set_ylim(0, max(12, baseline_n + 2))
    ax1.legend(title="Shift", fontsize=7, ncol=3)

    bool_labels = [v[1] for v in BOOLEAN_PARAMS.values()]
    bool_counts = [baseline_n + rng for _, rng in boolean_ranked]
    combo_labels = global_df["Global_shift"].tolist()
    combo_counts = global_df["Hit_count"].tolist()
    all_labels = [f"Not applied:\n{l}" for l in bool_labels] + [f"Global\n{l}" for l in combo_labels]
    all_counts = bool_counts + combo_counts
    colors = ["indianred"] * len(bool_labels) + ["steelblue"] * len(combo_labels)
    ax2.bar(range(len(all_labels)), all_counts, color=colors)
    ax2.axhline(baseline_n, color="red", linestyle="--", lw=1, alpha=0.6, label=f"Baseline ({baseline_n})")
    ax2.set_xticks(range(len(all_labels)))
    ax2.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Number of hit compounds")
    ax2.set_title("(B) ADMET liability filters (individual contribution, red) &\n"
                   "global stringency sweep (blue)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "SF_ADMET_Sensitivity_barplot.png")
    plt.savefig(fig1_path, dpi=600)
    plt.close()

    bool_impact_order = sorted(BOOLEAN_PARAMS.items(),
                                key=lambda kv: -abs([r for l, r in boolean_ranked if l == kv[1][1]][0]))
    running_params = {k: (BASELINE[k] if k in BASELINE else None) for k in BASELINE}
    drop_set = set(BOOLEAN_PARAMS.keys())  # start with none applied except continuous/ordinal
    cur = apply_filter(df, BASELINE, drop_bool=drop_set)
    funnel_labels.append("+ Lipinski/ADMET\ncontinuous+ordinal\ncutoffs")
    funnel_values.append(len(cur))
    funnel_after_physchem = len(cur)
    for key, (col, label) in bool_impact_order:
        drop_set = drop_set - {key}
        cur = apply_filter(df, BASELINE, drop_bool=drop_set)
        funnel_labels.append(f"+ {label}")
        funnel_values.append(len(cur))

    fig2, ax3 = plt.subplots(figsize=(9, 5.5))
    bars = ax3.bar(range(len(funnel_labels)), funnel_values, color="teal", edgecolor="black")
    for i, v in enumerate(funnel_values):
        ax3.text(i, v + max(funnel_values) * 0.02, str(v), ha="center", fontsize=10, fontweight="bold")
    ax3.set_xticks(range(len(funnel_labels)))
    ax3.set_xticklabels(funnel_labels, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Number of compounds remaining")
    ax3.set_title("Sequential ADMET Filtering Workflow\n(290 Pan-Active Compounds -> 3 Consensus Leads)")
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "SF_ADMET_FilterFunnel.png")
    plt.savefig(fig2_path, dpi=600)
    plt.close()

    print(f"\nSaved figures:\n  {fig1_path}\n  {fig2_path}")


if __name__ == "__main__":
    main()