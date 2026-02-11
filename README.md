# Towards Objective Gastrointestinal Auscultation:Automated Segmentation and Annotation of Bowel Sound Patterns  
(Event Detection + Automatic Pattern Labeling)

This is the official code repository associated with the paper:

📄 **Towards Objective Gastrointestinal Auscultation:Automated Segmentation and Annotation of Bowel Sound Patterns **
✍️ *Zahra Mansour, Verena Uslar, Dirk Weyhe, Danilo Hollosi and Nils Strodthoff.*
📅 **Currently under review**
This repository provides an end-to-end AI-based bowel sound (BS) auto-annotation pipeline for abdominal auscultation recordings.  
The system performs automatic bowel sound event detection and pattern classification to generate structured annotation outputs suitable for clinical research and AI-based clinical decision support development.
---

# Overview

The pipeline performs:

1. Automatic bowel sound event detection (segmentation)
2. AI-based classification of detected segments
3. Structured export of annotations (CSV / TXT)
4. Optional evaluation against ground truth annotations

It is designed for:

- Large-scale bowel sound dataset annotation
- Clinical gastrointestinal research
- Development of AI-based clinical decision-making systems
- Reproducible biomedical signal processing workflows

---

# Main Features

## 1. Automatic Event Detection

Supported segmentation methods (configurable in `config.yml`):

- Energy-based detection
- Modified energy detection
- RMS-based detection (recommended baseline)
- Sound Event Detection (SED) backend (optional)
- ATST-SED backend (optional)
- YAMNet backend (optional)
- Voice Activity Detection (VAD)

Post-processing options include:
- Minimum duration filtering
- Gap merging
- Adaptive thresholding

---

## 2. Automatic Pattern Classification

Each detected bowel sound segment is automatically assigned a predicted class label using a pretrained AI model (e.g., AST-based architecture or ensemble model).

Example Bowel sound patterns:

- SB  (Single Burst)
- MB  (Multiple Burst)
- CRS (Continuous Random Sound)
- HS  (Harmonic Sound)
- NONE / scilent


---

# Installation

It is recommended to create a clean Python environment.

## Install dependencies

install typical dependencies:

```bash
pip install numpy pandas scipy librosa torch transformers safetensors pyyaml
```

If using GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# Quick Start

## Step 1 – Configure the pipeline

Edit the configuration file:

```
config.yml
```

Set:

- Input paths
- Segmentation method
- Model checkpoint path
- Output directories
- Audio parameters

---

## Step 2 – Run the pipeline

```bash
python main.py
```

The system will:

1. Load audio recordings
2. Perform bowel sound segmentation
3. Predict class labels for detected segments
4. Export structured annotation files

---

# Input Data Format

Two input options are supported.

---

## Option A – Master CSV (Recommended)

Provide a CSV file with at least:

| Column Name | Description |
|-------------|------------|
| Patient     | Patient identifier |
| Wav_path    | Path to .wav file |

Example:

```csv
Patient,Wav_path
100425001,/path/to/audio/file1.wav
100425002,/path/to/audio/file2.wav
```

---

## Option B – Folder-Based Input

Place all `.wav` files in:

```
input_folder/
```

Patient IDs may be extracted automatically from filenames.

---

# Configuration Parameters (config.yml)

## Input Section

- input.master_csv
- input.input_folder
- input.patient_filter

---

## Audio Section

- audio.target_sr (e.g., 16000 Hz)
- audio.n_fft
- audio.hop_length
- audio.min_segment_duration
- audio.fixed_segment_length

---

## Segmentation Section

- segmentation.method  
  Options:  
  `energy | energy_mod | rms_mod | sed | atst | yamnet | vad`

- segmentation.merge.max_gap_sec
- segmentation.merge.min_duration_sec

---

## Filtering Section (Optional)

- filtering.mode  
  Options:  
  `none | heart | lung | default`

---

## Model Section

- model.mode (single | ensemble)
- model.device (cpu | cuda | auto)
- model.checkpoint_path
- model.ensemble_paths (if applicable)

---

## Evaluation Section (Optional)

- evaluation.enabled (true/false)
- evaluation.match_mode (patient | pair | basename)

---

# Output Files

The pipeline generates:

## 1. Annotation CSV

Typical format:

- Patient
- Wav_path
- Start_time_sec
- End_time_sec
- Predicted_Label

---

## 2. TXT Annotation File (Optional)

Format:

```
start_time    end_time    label
```

---

## 3. Evaluation Metrics (If Enabled)

- Sensitivity
- Specificity
- Precision
- F1-score
- ROC metrics

---


