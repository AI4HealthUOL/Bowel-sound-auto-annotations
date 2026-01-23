from __future__ import annotations

import os
from typing import Optional, List, Tuple


def ensure_dirs(output_folder: str, results_dir: str) -> None:
    """Create output folders if they do not exist."""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)


def normalize_path(p: str) -> str:
    """Normalize a file path for stable CSV outputs across OSes."""
    return os.path.abspath(str(p)).replace("\\", "/").strip()


def parse_patient_id(filename_stem: str) -> str:
    """Extract patient ID from file stem (prefix before first underscore)."""
    return filename_stem.split("_")[0] if "_" in filename_stem else filename_stem


def collect_wavs_from_csv(csv_path: str, allowed_patients: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    Read WAV paths from a CSV containing: Patient, Wav_path
    and optionally filter by patient IDs.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = df.columns.map(str).str.strip()
    if "Patient" not in df.columns or "Wav_path" not in df.columns:
        raise ValueError("MASTER_CSV must contain columns: 'Patient' and 'Wav_path'")

    df["Wav_path"] = df["Wav_path"].astype(str).str.replace("\\", "/", regex=False).str.strip()

    def normalize_patient(val: str) -> str:
        parts = str(val).split("_", 1)
        if len(parts) == 2:
            prefix, suffix = parts
            if suffix == "C":
                return val
            return prefix
        return val

    df["Patient"] = df["Patient"].apply(normalize_patient)

    if allowed_patients:
        df = df[df["Patient"].astype(str).isin(allowed_patients)].copy()

    df = df.sort_values(["Wav_path"]).drop_duplicates(subset=["Wav_path"], keep="first")

    pairs = []
    for _, row in df.iterrows():
        wav = normalize_path(row["Wav_path"])
        pat = str(row["Patient"]).strip()
        pairs.append((wav, pat))
    return pairs


def collect_wavs_from_folder(folder: str, allowed_patients: Optional[List[str]]) -> List[Tuple[str, str]]:
    """Collect *.wav files from a folder and derive patient IDs from filenames."""
    wavs: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".wav"):
            continue
        base = os.path.splitext(fname)[0]
        patient = parse_patient_id(base)
        if allowed_patients and patient not in allowed_patients:
            continue
        wavs.append((normalize_path(os.path.join(folder, fname)), patient))
    return wavs


def write_csv(rows: List[Tuple[str, str, float, float, int]], out_path: str) -> None:
    """Write segment rows into a consistent CSV schema."""
    import csv

    header = ["Patient", "Wav_path", "Start", "End", "Predicted_Label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_labeled_txt(segments, labels, out_path: str, decimals: int = 3) -> None:
    """
    Simple TXT format for quick inspection:
      start<TAB>end<TAB>label
    """
    assert len(segments) == len(labels), "segments and labels length mismatch"
    with open(out_path, "w", encoding="utf-8") as f:
        for (st, en), lb in zip(segments, labels):
            f.write(f"{st:.{decimals}f}\t{en:.{decimals}f}\t{int(lb)}\n")
