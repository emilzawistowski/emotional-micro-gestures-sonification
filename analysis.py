

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["condition"]      = df["condition"].str.strip().str.lower()
    df["identification"] = df["identification"].str.strip().str.lower()
    df["correct"]        = (df["condition"] == df["identification"]).astype(int)
    print(f"Loaded {len(df)} trials, {df['participant_id'].nunique()} participants.")
    return df


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def run(df: pd.DataFrame) -> None:

    # ── Accuracy ────────────────────────────────────────────────────────────
    n_correct = int(df["correct"].sum())
    n_total   = len(df)
    accuracy  = n_correct / n_total
    binom     = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    ci        = binom.proportion_ci(0.95)

    print("\n=== Identification Accuracy ===")
    print(f"  {n_correct}/{n_total} = {accuracy:.1%}")
    print(f"  Binomial p = {binom.pvalue:.4f}  {'*' if binom.pvalue < 0.05 else 'n.s.'}")
    print(f"  95% CI [{ci.low:.3f}, {ci.high:.3f}]")
    for cond in ["stress", "relax"]:
        sub = df[df["condition"] == cond]
        print(f"  {cond}: {sub['correct'].mean():.1%}  (n={len(sub)})")

    # ── SAM Arousal ─────────────────────────────────────────────────────────
    s_ar = df[df["condition"] == "stress"]["arousal"].values
    r_ar = df[df["condition"] == "relax"]["arousal"].values
    u_ar, p_ar = stats.mannwhitneyu(s_ar, r_ar, alternative="two-sided")
    d_ar = cohens_d(s_ar, r_ar)
    z_ar = stats.norm.ppf(1 - p_ar / 2)
    r_ar_eff = float(z_ar / np.sqrt(len(s_ar) + len(r_ar)))

    print("\n=== SAM Arousal ===")
    print(f"  Stress M={s_ar.mean():.2f} SD={s_ar.std():.2f} Mdn={np.median(s_ar):.1f}")
    print(f"  Relax  M={r_ar.mean():.2f} SD={r_ar.std():.2f} Mdn={np.median(r_ar):.1f}")
    print(f"  U={u_ar:.0f}  p={p_ar:.4f}  r={r_ar_eff:.3f}  d={d_ar:.3f} ({interpret_d(d_ar)})")

    # ── SAM Valence ─────────────────────────────────────────────────────────
    s_va = df[df["condition"] == "stress"]["valence"].values
    r_va = df[df["condition"] == "relax"]["valence"].values
    u_va, p_va = stats.mannwhitneyu(s_va, r_va, alternative="two-sided")
    d_va = cohens_d(s_va, r_va)
    z_va = stats.norm.ppf(1 - p_va / 2)
    r_va_eff = float(z_va / np.sqrt(len(s_va) + len(r_va)))

    print("\n=== SAM Valence ===")
    print(f"  Stress M={s_va.mean():.2f} SD={s_va.std():.2f} Mdn={np.median(s_va):.1f}")
    print(f"  Relax  M={r_va.mean():.2f} SD={r_va.std():.2f} Mdn={np.median(r_va):.1f}")
    print(f"  U={u_va:.0f}  p={p_va:.4f}  r={r_va_eff:.3f}  d={d_va:.3f} ({interpret_d(d_va)})")

    # ── Confidence ──────────────────────────────────────────────────────────
    rho, p_rho = stats.spearmanr(df["confidence"], df["correct"])
    print(f"\n=== Confidence as Moderator ===")
    print(f"  Spearman rho={rho:.3f}  p={p_rho:.4f}  {'*' if p_rho < 0.05 else 'n.s.'}")
    for lv in sorted(df["confidence"].unique()):
        sub = df[df["confidence"] == lv]
        print(f"  Confidence {lv}: accuracy = {sub['correct'].mean():.1%}  (n={len(sub)})")

    # ── Per participant ──────────────────────────────────────────────────────
    pp = (df.groupby("participant_id")
            .agg(n=("correct","count"),
                 n_correct=("correct","sum"),
                 accuracy=("correct","mean"),
                 mean_arousal=("arousal","mean"),
                 mean_valence=("valence","mean"),
                 mean_conf=("confidence","mean"))
            .reset_index())
    print(f"\n=== Per Participant ===")
    print(pp.to_string(index=False))

    # ── APA sentences ────────────────────────────────────────────────────────
    print("\n=== APA sentences ===")
    print(f"Listeners identified the emotional condition in {accuracy:.1%} of trials "
          f"(n={n_total}), significantly above chance (50%), binomial p={binom.pvalue:.3f}, "
          f"95% CI [{ci.low:.2f}, {ci.high:.2f}].")
    print(f"SAM Arousal: stress (M={s_ar.mean():.2f}, SD={s_ar.std():.2f}, Mdn={np.median(s_ar):.1f}) > "
          f"relax (M={r_ar.mean():.2f}, SD={r_ar.std():.2f}, Mdn={np.median(r_ar):.1f}), "
          f"U={u_ar:.0f}, p={p_ar:.3f}, d={d_ar:.2f} ({interpret_d(d_ar)}).")
    print(f"SAM Valence: stress (M={s_va.mean():.2f}, SD={s_va.std():.2f}, Mdn={np.median(s_va):.1f}) < "
          f"relax (M={r_va.mean():.2f}, SD={r_va.std():.2f}, Mdn={np.median(r_va):.1f}), "
          f"U={u_va:.0f}, p={p_va:.3f}, d={d_va:.2f} ({interpret_d(d_va)}).")
    print(f"Confidence correlated with accuracy: rho={rho:.3f}, p={p_rho:.3f}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to results.csv")
    args = p.parse_args()
    df = load(args.data)
    run(df)


if __name__ == "__main__":
    main()