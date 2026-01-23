from __future__ import annotations

import os
from math import ceil
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CLASSES = [0, 1, 2, 3, 4]

CLASS_NAMES = {
    0: "Silent",
    1: "SB",
    2: "MB",
    3: "CRS",
    4: "HS",
}

CLASS_COLORS = [
    "#1f77b4",  # Silent
    "#ff7f0e",  # SB
    "#2ca02c",  # MB
    "#d62728",  # CRS
    "#9467bd",  # HS
]


def class_names_list() -> List[str]:
    return [CLASS_NAMES[c] for c in CLASSES]


def class_colors_list() -> List[str]:
    return [CLASS_COLORS[i] for i in range(len(CLASSES))]


# ---------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def intervals_total_duration(intervals):
    return sum(max(0.0, e - s) for s, e in intervals)


def intersection_duration(intervals_a, intervals_b):
    A = merge_intervals(intervals_a)
    B = merge_intervals(intervals_b)
    i = j = 0
    inter = 0.0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if s < e:
            inter += (e - s)
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return inter


# ---------------------------------------------------------------------
# Patient normalization (to match your original behavior)
# ---------------------------------------------------------------------

def normalize_patient_name_for_original(val: str) -> str:
    if pd.isna(val):
        return val
    s = str(val).strip()
    if "_" in s:
        base, _, rest = s.partition("_")
        if rest == "C":
            return s
        return base
    return s


# ---------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------

def _clean_basic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df.dropna(subset=["Patient", "Wav_path", "Start", "End"]).copy()
    df = df[df["End"] > df["Start"]].copy()
    df["Patient"] = df["Patient"].astype(str).str.strip()
    df["Wav_path"] = df["Wav_path"].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    df = df[df[label_col].isin(CLASSES)].copy()
    return df


def read_and_filter(pred_path: str, orig_path: str, match_mode: str = "patient") -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_path)
    orig = pd.read_csv(orig_path)

    if "Patient" in orig.columns:
        orig["Patient"] = orig["Patient"].apply(normalize_patient_name_for_original)

    pred = _clean_basic(pred, "Predicted_Label")
    orig = _clean_basic(orig, "Label")

    keep_patients = set(pred["Patient"].unique())
    orig = orig[orig["Patient"].isin(keep_patients)].copy()

    if match_mode not in {"patient", "pair", "basename"}:
        match_mode = "patient"

    if match_mode in {"pair", "basename"}:
        if match_mode == "basename":
            pred["_key_wav"] = pred["Wav_path"].map(os.path.basename)
            orig["_key_wav"] = orig["Wav_path"].map(os.path.basename)
        else:
            pred["_key_wav"] = pred["Wav_path"]
            orig["_key_wav"] = orig["Wav_path"]

        keep_pairs = set(
            pred[["Patient", "_key_wav"]].drop_duplicates().itertuples(index=False, name=None)
        )
        orig["_pair"] = list(zip(orig["Patient"], orig["_key_wav"]))
        orig = orig[orig["_pair"].isin(keep_pairs)].copy()
        orig.drop(columns=["_pair", "_key_wav"], inplace=True)
        pred.drop(columns=["_key_wav"], inplace=True, errors="ignore")

    print(f"[INFO] match_mode={match_mode}")
    print(f"[INFO] Pred rows: {len(pred)}, Patients: {pred['Patient'].nunique()}")
    print(f"[INFO] Orig rows after filter: {len(orig)}, Patients: {orig['Patient'].nunique()}")
    if len(orig) == 0:
        print("[WARN] Original set is empty after filtering. Try --match patient (default) or --match basename if paths differ.")

    return pred.reset_index(drop=True), orig.reset_index(drop=True)


def detect_label_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override and override in df.columns:
        return override
    candidates = ["Auto_Label", "AutoLabel", "Label", "Predicted_Label", "Prediction"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect a label column in auto CSV. Tried: {candidates}. Columns: {list(df.columns)}")


def read_and_filter_auto(auto_path: str, pred_df: pd.DataFrame, match_mode: str = "patient", auto_label_col: Optional[str] = None):
    auto = pd.read_csv(auto_path)
    label_col = detect_label_col(auto, auto_label_col)
    auto = _clean_basic(auto, label_col)
    print(f"[INFO] Auto rows kept (no filtering): {len(auto)}, Patients: {auto['Patient'].nunique()} (label col='{label_col}')")
    return auto.reset_index(drop=True), label_col


# ---------------------------------------------------------------------
# Distributions & counts
# ---------------------------------------------------------------------

def class_distribution(df: pd.DataFrame, label_col: str):
    counts = df[label_col].value_counts().reindex(CLASSES, fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100.0) if total > 0 else counts.astype(float)
    return counts, pct


def class_distribution_by_subject(df: pd.DataFrame, label_col: str):
    tbl = (
        df.groupby(["Patient", label_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
    )
    pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    return tbl, pct


def macro_avg_percent(pct_by_subject_df: pd.DataFrame) -> pd.Series:
    valid = pct_by_subject_df[pct_by_subject_df.sum(axis=1) > 0]
    if valid.empty:
        return pd.Series(0.0, index=CLASSES, dtype="float")
    return valid.mean(axis=0, skipna=True).reindex(CLASSES, fill_value=0.0)


def macro_counts_mean_and_sem(counts_by_subject_df: pd.DataFrame):
    if counts_by_subject_df.empty:
        zero = pd.Series(0.0, index=CLASSES, dtype="float")
        return zero, zero
    valid = counts_by_subject_df[counts_by_subject_df.sum(axis=1) > 0]
    if valid.empty:
        zero = pd.Series(0.0, index=CLASSES, dtype="float")
        return zero, zero
    mean = valid.mean(axis=0).reindex(CLASSES, fill_value=0.0)
    n = max(len(valid), 1)
    std = valid.std(axis=0, ddof=1).fillna(0.0).reindex(CLASSES, fill_value=0.0)
    sem = std / np.sqrt(n)
    return mean, sem


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _bar3_colored(ax, x, w, a, b, c, labels=("Original", "Predicted", "AutoLabelled")):
    colors = class_colors_list()
    for i, col in enumerate(colors):
        ax.bar(x[i] - w, a[i], width=w, color=col, alpha=0.95)
        ax.bar(x[i],     b[i], width=w, color=col, alpha=0.95)
        ax.bar(x[i] + w, c[i], width=w, color=col, alpha=0.95)

    ax.bar([], [], color="white", label=labels[0])
    ax.bar([], [], color="white", label=labels[1])
    ax.bar([], [], color="white", label=labels[2])


def plot_class_distribution_overall3(pct_orig, pct_pred, pct_auto, out_path, title_suffix="(All subjects, macro-avg)"):
    x = np.arange(len(CLASSES))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar3_colored(ax, x, w, pct_orig.values, pct_pred.values, pct_auto.values)
    ax.set_xticks(x, class_names_list())
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Class")
    ax.set_title(f"Class Distribution {title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_distribution_per_subject3(pct_orig_subj, pct_pred_subj, pct_auto_subj, out_path):
    subjects = sorted(set(pct_orig_subj.index).union(pct_pred_subj.index).union(pct_auto_subj.index))
    n = len(subjects)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    x = np.arange(len(CLASSES))

    for i, subj in enumerate(subjects):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        o = pct_orig_subj.loc[subj] if subj in pct_orig_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        p = pct_pred_subj.loc[subj] if subj in pct_pred_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        a = pct_auto_subj.loc[subj] if subj in pct_auto_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        _bar3_colored(
            ax, x, 0.22,
            o.reindex(CLASSES, fill_value=0).values,
            p.reindex(CLASSES, fill_value=0).values,
            a.reindex(CLASSES, fill_value=0).values
        )
        ax.set_xticks(x, class_names_list())
        ax.set_ylim(0, 100)
        ax.set_title(f"Patient: {subj}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if r == 0 and c == 0:
            ax.legend()

    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle("Class Distribution per Subject (Percentages)", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_distribution_subplots_per_subject3(pct_orig_subj, pct_pred_subj, pct_auto_subj, out_path):
    """
    Per-subject grid where each ROW corresponds to a subject and columns are:
      Original | Predicted | AutoLabelled
    """
    subjects = sorted(set(pct_orig_subj.index).union(pct_pred_subj.index).union(pct_auto_subj.index))
    n = len(subjects)
    if n == 0:
        return

    ncols = 3
    nrows = n
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3*nrows), squeeze=False)

    colors = class_colors_list()
    names = class_names_list()
    x = np.arange(len(CLASSES))

    for i, subj in enumerate(subjects):
        # Original
        ax = axes[i][0]
        o = pct_orig_subj.loc[subj] if subj in pct_orig_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        ax.bar(x, o.reindex(CLASSES, fill_value=0).values, color=colors)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 100)
        if i == 0:
            ax.set_title("Original")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Predicted
        ax = axes[i][1]
        p = pct_pred_subj.loc[subj] if subj in pct_pred_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        ax.bar(x, p.reindex(CLASSES, fill_value=0).values, color=colors)
        ax.set_xticks(x, names)
        if i == 0:
            ax.set_title("Predicted")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Auto
        ax = axes[i][2]
        a = pct_auto_subj.loc[subj] if subj in pct_auto_subj.index else pd.Series(0, index=CLASSES, dtype=float)
        ax.bar(x, a.reindex(CLASSES, fill_value=0).values, color=colors)
        ax.set_xticks(x, names)
        if i == 0:
            ax.set_title("AutoLabelled")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[i][0].set_ylabel(f"{subj}")

    plt.tight_layout()
    fig.suptitle("Class Distribution per Subject (pooled %)  Original / Predicted / Auto", y=0.995, fontsize=14)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_counts_overall3(counts_orig, counts_pred, counts_auto, out_path, title_suffix="(TOTAL counts; affected by #subjects)"):
    x = np.arange(len(CLASSES))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar3_colored(
        ax, x, w,
        counts_orig.reindex(CLASSES, fill_value=0).values,
        counts_pred.reindex(CLASSES, fill_value=0).values,
        counts_auto.reindex(CLASSES, fill_value=0).values,
    )
    ax.set_xticks(x, class_names_list())
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.set_title(f"Class Counts {title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_counts_macro_overall3(mean_o, mean_p, mean_a, out_path, sem_o=None, sem_p=None, sem_a=None,
                                     title="Class Counts (macro-averaged: mean per subject)"):
    x = np.arange(len(CLASSES))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))
    y1 = mean_o.values
    y2 = mean_p.values
    y3 = mean_a.values
    e1 = sem_o.values if sem_o is not None else None
    e2 = sem_p.values if sem_p is not None else None
    e3 = sem_a.values if sem_a is not None else None

    colors = class_colors_list()
    for i, col in enumerate(colors):
        ax.bar(x[i] - w, y1[i], width=w, color=col, label="Original" if i == 0 else "", alpha=0.95,
               yerr=(e1[i] if e1 is not None else None), capsize=3 if e1 is not None else None)
        ax.bar(x[i], y2[i], width=w, color=col, label="Predicted" if i == 0 else "", alpha=0.95,
               yerr=(e2[i] if e2 is not None else None), capsize=3 if e2 is not None else None)
        ax.bar(x[i] + w, y3[i], width=w, color=col, label="AutoLabelled" if i == 0 else "", alpha=0.95,
               yerr=(e3[i] if e3 is not None else None), capsize=3 if e3 is not None else None)

    ax.set_xticks(x, class_names_list())
    ax.set_ylabel("Avg count per subject")
    ax.set_xlabel("Class")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Duration"] = x["End"] - x["Start"]
    return x


def durations_by_class(df: pd.DataFrame, label_col: str):
    x = add_duration(df)
    return {cls: x.loc[x[label_col] == cls, "Duration"].values for cls in CLASSES}


def durations_by_class_per_subject(df: pd.DataFrame, label_col: str):
    x = add_duration(df)
    out = {}
    for subj, subdf in x.groupby("Patient"):
        out[subj] = {cls: subdf.loc[subdf[label_col] == cls, "Duration"].values for cls in CLASSES}
    return out


def plot_duration_boxplots_overall3(dur_orig, dur_pred, dur_auto, out_path,
                                    title="Duration per Class (seconds) Original vs Predicted vs AutoLabelled"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    class_colors = class_colors_list()
    class_names = class_names_list()

    for ax, data_map, title_lbl in zip(
        axes,
        [dur_orig, dur_pred, dur_auto],
        ["Original", "Predicted", "AutoLabelled"],
    ):
        data = [data_map[c] if len(data_map[c]) else np.array([np.nan]) for c in CLASSES]
        bp = ax.boxplot(data, positions=np.arange(len(CLASSES)), patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=class_colors[i], edgecolor="black")
        for m in bp["medians"]:
            m.set(color="black")
        ax.set_xticks(np.arange(len(CLASSES)), class_names)
        ax.set_title(title_lbl)
        ax.set_xlabel("Class")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Duration (s)")
    fig.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_distribution_and_duration_combined3(pct_o, pct_p, pct_a, dur_o, dur_p, dur_a, out_path,
                                             title="Distribution (top) and Duration (bottom) - Original | Predicted | AutoLabelled"):
    class_names = class_names_list()
    class_colors = class_colors_list()
    x = np.arange(len(CLASSES))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

    # Top row: pooled distributions
    for col_idx, (vals, title_lbl) in enumerate(zip(
        [pct_o, pct_p, pct_a],
        ["Original - Distribution (%)", "Predicted - Distribution (%)", "AutoLabelled - Distribution (%)"],
    )):
        ax = axes[0, col_idx]
        v = vals.reindex(CLASSES, fill_value=0).values
        ax.bar(x, v, color=class_colors)
        ax.set_xticks(x, class_names)
        ax.set_title(title_lbl)
        ax.set_ylim(0, 105)
        if col_idx == 0:
            ax.set_ylabel("Percentage (%)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Bottom row: durations
    data_o = [dur_o[c] if len(dur_o[c]) else np.array([np.nan]) for c in CLASSES]
    data_p = [dur_p[c] if len(dur_p[c]) else np.array([np.nan]) for c in CLASSES]
    data_a = [dur_a[c] if len(dur_a[c]) else np.array([np.nan]) for c in CLASSES]

    for col_idx, (data, title_lbl) in enumerate(zip(
        [data_o, data_p, data_a],
        ["Original - Duration", "Predicted - Duration", "AutoLabelled - Duration"],
    )):
        ax = axes[1, col_idx]
        bp = ax.boxplot(data, positions=np.arange(len(CLASSES)), patch_artist=True, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set(facecolor=class_colors[i], edgecolor="black")
        for m in bp["medians"]:
            m.set(color="black")
        ax.set_xticks(np.arange(len(CLASSES)), class_names)
        ax.set_title(title_lbl)
        if col_idx == 0:
            ax.set_ylabel("Duration (s)")
        ax.set_ylim(0, 8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=CLASS_NAMES[c],
                   markerfacecolor=class_colors[i], markersize=10)
        for i, c in enumerate(CLASSES)
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.95))

    fig.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 0.94, 0.96])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# IoU + Coverage
# ---------------------------------------------------------------------

def iou_per_subject_and_class(pred: pd.DataFrame, orig: pd.DataFrame, match_mode="patient"):
    subjects = sorted(pred["Patient"].unique())
    use_basename = (match_mode == "basename")

    pred_key_wav = pred["Wav_path"].map(os.path.basename) if use_basename else pred["Wav_path"]
    orig_key_wav = orig["Wav_path"].map(os.path.basename) if use_basename else orig["Wav_path"]

    pred_tmp = pred.copy()
    pred_tmp["_key_wav"] = pred_key_wav
    orig_tmp = orig.copy()
    orig_tmp["_key_wav"] = orig_key_wav

    total_inter = 0.0
    total_union = 0.0
    class_inter = {cls: 0.0 for cls in CLASSES}
    class_union = {cls: 0.0 for cls in CLASSES}
    subj_inter = {s: {cls: 0.0 for cls in CLASSES} for s in subjects}
    subj_union = {s: {cls: 0.0 for cls in CLASSES} for s in subjects}

    for (patient, wav), pred_wav in pred_tmp.groupby(["Patient", "_key_wav"]):
        orig_wav = orig_tmp[(orig_tmp["Patient"] == patient) & (orig_tmp["_key_wav"] == wav)]
        for cls in CLASSES:
            p_rows = pred_wav[pred_wav["Predicted_Label"] == cls][["Start", "End"]].to_numpy()
            o_rows = orig_wav[orig_wav["Label"] == cls][["Start", "End"]].to_numpy()

            p_ints = [(float(s), float(e)) for s, e in p_rows]
            o_ints = [(float(s), float(e)) for s, e in o_rows]

            inter = intersection_duration(p_ints, o_ints)
            dur_p = intervals_total_duration(p_ints)
            dur_o = intervals_total_duration(o_ints)
            uni = dur_p + dur_o - inter

            subj_inter[patient][cls] += inter
            subj_union[patient][cls] += uni
            class_inter[cls] += inter
            class_union[cls] += uni
            total_inter += inter
            total_union += uni

    rows = {}
    for s in subjects:
        rows[s] = {
            cls: (subj_inter[s][cls] / subj_union[s][cls]) if subj_union[s][cls] > 0 else np.nan
            for cls in CLASSES
        }
    subj_class_iou = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=CLASSES)
    class_iou_overall = pd.Series({
        cls: (class_inter[cls] / class_union[cls]) if class_union[cls] > 0 else np.nan
        for cls in CLASSES
    })
    micro_iou_overall = (total_inter / total_union) if total_union > 0 else np.nan

    return subj_class_iou, class_iou_overall, micro_iou_overall


def coverage_per_subject_and_class(pred: pd.DataFrame, orig: pd.DataFrame, match_mode="patient"):
    """
    Coverage_{s,c} = intersection_duration(pred_c, orig_c) / total_orig_duration_c
    """
    subjects = sorted(pred["Patient"].unique())
    use_basename = (match_mode == "basename")

    pred_key_wav = pred["Wav_path"].map(os.path.basename) if use_basename else pred["Wav_path"]
    orig_key_wav = orig["Wav_path"].map(os.path.basename) if use_basename else orig["Wav_path"]

    pred_tmp = pred.copy()
    pred_tmp["_key_wav"] = pred_key_wav
    orig_tmp = orig.copy()
    orig_tmp["_key_wav"] = orig_key_wav

    total_inter = 0.0
    total_orig = 0.0
    class_inter = {cls: 0.0 for cls in CLASSES}
    class_orig = {cls: 0.0 for cls in CLASSES}
    subj_inter = {s: {cls: 0.0 for cls in CLASSES} for s in subjects}
    subj_orig = {s: {cls: 0.0 for cls in CLASSES} for s in subjects}

    for (patient, wav), pred_wav in pred_tmp.groupby(["Patient", "_key_wav"]):
        orig_wav = orig_tmp[(orig_tmp["Patient"] == patient) & (orig_tmp["_key_wav"] == wav)]
        for cls in CLASSES:
            p_rows = pred_wav[pred_wav["Predicted_Label"] == cls][["Start", "End"]].to_numpy()
            o_rows = orig_wav[orig_wav["Label"] == cls][["Start", "End"]].to_numpy()

            p_ints = [(float(s), float(e)) for s, e in p_rows]
            o_ints = [(float(s), float(e)) for s, e in o_rows]

            inter = intersection_duration(p_ints, o_ints)
            dur_o = intervals_total_duration(o_ints)

            subj_inter[patient][cls] += inter
            subj_orig[patient][cls] += dur_o
            class_inter[cls] += inter
            class_orig[cls] += dur_o
            total_inter += inter
            total_orig += dur_o

    rows = {}
    for s in subjects:
        rows[s] = {
            cls: (subj_inter[s][cls] / subj_orig[s][cls]) if subj_orig[s][cls] > 0 else np.nan
            for cls in CLASSES
        }
    subj_class_cov = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=CLASSES)

    class_cov_overall = pd.Series({
        cls: (class_inter[cls] / class_orig[cls]) if class_orig[cls] > 0 else np.nan
        for cls in CLASSES
    })
    micro_cov_overall = (total_inter / total_orig) if total_orig > 0 else np.nan

    return subj_class_cov, class_cov_overall, micro_cov_overall


def plot_iou_overall(class_iou, micro_iou, out_path, title="IoU per Class All subjects"):
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = class_iou.reindex(CLASSES).values
    ax.bar(x, vals, color=class_colors_list())
    ax.set_xticks(x, class_names_list())
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.set_xlabel("Class")
    ttl = f"{title}\nMicro-averaged IoU: {micro_iou:.3f}" if not np.isnan(micro_iou) else title
    ax.set_title(ttl)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou_per_subject(subj_class_iou, out_path):
    subjects = list(subj_class_iou.index)
    n = len(subjects)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for i, subj in enumerate(subjects):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        vals = subj_class_iou.loc[subj].reindex(CLASSES).values
        ax.bar(np.arange(len(CLASSES)), np.nan_to_num(vals, nan=0.0), color=class_colors_list())
        ax.set_xticks(np.arange(len(CLASSES)), class_names_list())
        ax.set_ylim(0, 1)
        ax.set_title(f"Patient: {subj}")
        ax.set_ylabel("IoU")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    fig.suptitle("IoU per Class per Subject", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_evaluation(
    pred_path: str,
    orig_path: str,
    out_dir: str,
    match_mode: str = "patient",
    auto_path: Optional[str] = None,
    auto_label_col: Optional[str] = None,
) -> None:
    """
    Run the full evaluation suite.

    Outputs:
    - CSV summaries (distributions, IoU, coverage)
    - Plots (distributions, counts, durations, IoU)
    """
    os.makedirs(out_dir, exist_ok=True)

    pred, orig = read_and_filter(pred_path, orig_path, match_mode)

    auto = None
    auto_lbl_col = None
    if auto_path:
        auto, auto_lbl_col = read_and_filter_auto(auto_path, pred, match_mode, auto_label_col)

    # --- Distributions & counts ---
    counts_o, pct_o_pooled = class_distribution(orig, "Label")
    counts_p, pct_p_pooled = class_distribution(pred, "Predicted_Label")
    if auto is not None:
        counts_a, pct_a_pooled = class_distribution(auto, auto_lbl_col)
    else:
        counts_a = pd.Series(0, index=CLASSES, dtype="int64")
        pct_a_pooled = pd.Series(0.0, index=CLASSES, dtype="float")

    tbl_o, pct_o_subj = class_distribution_by_subject(orig, "Label")
    tbl_p, pct_p_subj = class_distribution_by_subject(pred, "Predicted_Label")
    if auto is not None:
        tbl_a, pct_a_subj = class_distribution_by_subject(auto, auto_lbl_col)
    else:
        pct_a_subj = pd.DataFrame(columns=CLASSES, dtype=float)
        tbl_a = pd.DataFrame(columns=CLASSES, dtype=int)

    pct_o_macro = macro_avg_percent(pct_o_subj)
    pct_p_macro = macro_avg_percent(pct_p_subj)
    pct_a_macro = macro_avg_percent(pct_a_subj)

    macro_o, sem_o = macro_counts_mean_and_sem(tbl_o)
    macro_p, sem_p = macro_counts_mean_and_sem(tbl_p)
    macro_a, sem_a = macro_counts_mean_and_sem(tbl_a)

    all_subjs = sorted(set(pct_o_subj.index).union(pct_p_subj.index).union(pct_a_subj.index))
    pct_o_subj_plot = pct_o_subj.reindex(all_subjs).reindex(columns=CLASSES, fill_value=0)
    pct_p_subj_plot = pct_p_subj.reindex(all_subjs).reindex(columns=CLASSES, fill_value=0)
    pct_a_subj_plot = pct_a_subj.reindex(all_subjs).reindex(columns=CLASSES, fill_value=0)

    pd.DataFrame({
        "Original_count_TOTAL": counts_o, "Original_pct_pooled": pct_o_pooled, "Original_pct_macro": pct_o_macro,
        "Pred_count_TOTAL":     counts_p, "Pred_pct_pooled":    pct_p_pooled,  "Pred_pct_macro":    pct_p_macro,
        "Auto_count_TOTAL":     counts_a, "Auto_pct_pooled":    pct_a_pooled,  "Auto_pct_macro":    pct_a_macro,
        "Original_macro_count_mean": macro_o,
        "Pred_macro_count_mean":     macro_p,
        "Auto_macro_count_mean":     macro_a,
        "Original_macro_count_sem":  sem_o,
        "Pred_macro_count_sem":      sem_p,
        "Auto_macro_count_sem":      sem_a,
    }).to_csv(os.path.join(out_dir, "class_distribution_overall.csv"), index_label="Class")

    pd.concat(
        {"Original_pct": pct_o_subj_plot, "Pred_pct": pct_p_subj_plot, "Auto_pct": pct_a_subj_plot},
        axis=1
    ).to_csv(os.path.join(out_dir, "class_distribution_per_subject_percent.csv"), index_label="Patient")

    # Macro distribution plots
    plot_class_distribution_overall3(
        pct_o_macro, pct_p_macro, pct_a_macro,
        os.path.join(out_dir, "class_distribution_overall_macro.png"),
        title_suffix="(All subjects, macro-averaged)",
    )
    plot_class_distribution_per_subject3(
        pct_o_subj_plot, pct_p_subj_plot, pct_a_subj_plot,
        os.path.join(out_dir, "class_distribution_per_subject.png"),
    )

    # Pooled distribution plots
    plot_class_distribution_overall3(
        pct_o_pooled, pct_p_pooled, pct_a_pooled,
        os.path.join(out_dir, "class_distribution_overall_pooled.png"),
        title_suffix="(All subjects, pooled %)",
    )
    plot_class_distribution_subplots_per_subject3(
        pct_o_subj_plot, pct_p_subj_plot, pct_a_subj_plot,
        os.path.join(out_dir, "class_distribution_subplots_per_subject.png"),
    )

    # Counts
    plot_class_counts_macro_overall3(
        macro_o, macro_p, macro_a,
        os.path.join(out_dir, "class_counts_overall_macro.png"),
        sem_o=sem_o, sem_p=sem_p, sem_a=sem_a,
    )
    plot_class_counts_overall3(
        counts_o, counts_p, counts_a,
        os.path.join(out_dir, "class_counts_overall_TOTALS.png"),
    )

    # Durations + combined figure (if auto provided)
    dur_o = durations_by_class(orig, "Label")
    dur_p = durations_by_class(pred, "Predicted_Label")
    if auto is not None:
        dur_a = durations_by_class(auto, auto_lbl_col)
        plot_duration_boxplots_overall3(
            dur_o, dur_p, dur_a,
            os.path.join(out_dir, "duration_boxplots_overall_3col.png"),
        )
        plot_distribution_and_duration_combined3(
            pct_o_pooled, pct_p_pooled, pct_a_pooled,
            dur_o, dur_p, dur_a,
            os.path.join(out_dir, "combined_distribution_duration_3x2.png"),
        )

    # IoU
    subj_class_iou, class_iou_overall, micro_iou = iou_per_subject_and_class(pred, orig, match_mode)
    subj_class_iou.to_csv(os.path.join(out_dir, "iou_per_subject_per_class.csv"), index_label="Patient")
    pd.DataFrame({"IoU": class_iou_overall}).to_csv(os.path.join(out_dir, "iou_overall_per_class.csv"), index_label="Class")
    with open(os.path.join(out_dir, "iou_micro_overall.txt"), "w") as f:
        f.write(f"Micro-averaged IoU across all classes & subjects: {micro_iou:.6f}\n")

    plot_iou_overall(class_iou_overall, micro_iou, os.path.join(out_dir, "iou_overall_per_class.png"))
    plot_iou_per_subject(subj_class_iou, os.path.join(out_dir, "iou_per_subject_per_class.png"))

    # Coverage
    subj_class_cov, class_cov_overall, micro_cov = coverage_per_subject_and_class(pred, orig, match_mode)
    subj_class_cov.to_csv(os.path.join(out_dir, "coverage_per_subject_per_class.csv"), index_label="Patient")

    cov_macro = subj_class_cov.mean(axis=0, skipna=True).reindex(CLASSES)
    cov_overall_df = pd.DataFrame({
        "Coverage_weighted": class_cov_overall,
        "Coverage_macro_subject_mean": cov_macro,
    })
    cov_overall_df.to_csv(os.path.join(out_dir, "coverage_overall_per_class.csv"), index_label="Class")

    with open(os.path.join(out_dir, "coverage_micro_overall.txt"), "w") as f:
        f.write(f"Micro-averaged coverage across all classes & subjects: {micro_cov:.6f}\n")

    print("Done.")
    print(f"Saved outputs to: {os.path.abspath(out_dir)}")
