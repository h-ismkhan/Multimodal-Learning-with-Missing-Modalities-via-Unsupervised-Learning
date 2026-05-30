In the name of God

# UL4M4 on MM-IMDb: Cluster-Guided Iterative Imputation for Missing Modalities

This repository contains the implementation of **UL4M4** applied to the **MM-IMDb** multi-label movie genre classification benchmark. UL4M4 handles missing modalities at both training and inference time through a two-stage unsupervised approach — multi-modal clustering followed by iterative greedy completion — with no encoder fine-tuning required. For full method details, please refer to the accompanying paper.

---

## Repository Structure

```
.
├── mmimdb_loader.py      ← Dataset class, modality masking, collate function
├── ul4m4-mmimdb.py       ← Full UL4M4 pipeline (embedding → clustering → imputation → training)
└── README.md
```

---

## Dataset

### What is MM-IMDb?

MM-IMDb is a multi-modal dataset for **multi-label movie genre classification** across **27 genres** (e.g. Drama, Comedy, Thriller, Romance). Each sample contains:

- A **movie poster** (image modality)
- A **plot summary** (text modality)
- One or more **genre labels** (multi-label target)

### Expected Folder Structure

```
mmimdb/
├── split.json              ← train / dev / test sample ID lists
└── dataset/
    ├── <sample_id>.json    ← metadata: genres, plot, title, …
    ├── <sample_id>.jpeg    ← movie poster image
    ├── ...
```

`split.json` must contain keys `"train"`, `"dev"`, and `"test"`, each mapping to a list of sample ID strings that correspond to `<sample_id>.json` and `<sample_id>.jpeg` files inside the `dataset/` subdirectory.

---

## Data Preprocessing

### Step 1 — Dataset Loading (`mmimdb_loader.py`)

`MMIMDbDatasetCLIP` reads raw images and plot text per sample. For each item it:

1. Loads `<sample_id>.json` to extract the plot summary (first element of the `plot` list) and the list of genres.
2. Constructs a **binary multi-label vector** of length 27 (one entry per genre).
3. Loads `<sample_id>.jpeg`, converts to RGB, resizes to 224×224, and applies standard CLIP ImageNet normalisation (`mean=[0.48145466, 0.4578275, 0.40821073]`, `std=[0.26862954, 0.26130258, 0.27577711]`).
4. Applies **deterministic modality masking** based on the chosen missing configuration and a fixed random seed: missing images are replaced with zero tensors; missing text is replaced with an empty string `""`.

### Step 2 — CLIP Embedding Extraction (`ul4m4-mmimdb.py`)

Frozen CLIP ViT-B/32 encoders produce:

- **Image embeddings**: `[N, 512]` — zeroed out for missing images before storage.
- **Text embeddings**: `[N, 512]` — zeroed out for missing text before storage.

Modality availability masks (`has_image`, `has_text`) are preserved alongside the embeddings for all subsequent steps.

### Step 3 — Normalisation (training set statistics only)

Per-modality mean and standard deviation are computed **exclusively over available (non-missing) training samples** and then applied to normalise all three splits (train / dev / test). Entries corresponding to missing modalities remain zero and are excluded from distance computations via the availability masks.

### Step 4 — Stage 1: Partial-Modality K-Means Clustering

K-means clustering is run on the normalised training embeddings using a partial-modality distance that only compares modalities present in **both** a sample and a cluster centre, normalised by the number of shared modalities and each modality's embedding dimensionality. Pairs sharing no modalities are assigned infinite distance.

Cluster centres are initialised via sklearn's k-means++ (on concatenated embeddings for seed selection only), then mapped back to the nearest actual training sample so that centres inherit real, potentially partial, modalities. The EM loop (assignment → update) then uses the correct partial-modality distance for up to `--kmeans-iters` iterations or until convergence. Centre updates compute the mean over all cluster members that **possess** each modality; if no member possesses a modality, that modality remains absent in the centre.

### Step 5 — Stage 2: Iterative Greedy Completion

For every incomplete sample (missing at least one modality) in all three splits:

1. **Initialise $k$ candidates**: each candidate copies the sample's available modalities and fills each missing modality from the corresponding cluster centre (if the centre has it).
2. **Greedy loop**: compute partial-modality distances between all remaining candidates and all cluster centres; select the globally best (candidate, centre) pair; remove that candidate from the active set; copy any modalities it provides that the sample still lacks. Repeat until the sample is fully complete.

Dev and test splits use cluster centres and normalisation statistics derived **solely from the training set**.

### Step 6 — De-normalisation

After completion, all embeddings (including imputed ones) are de-normalised back to the original CLIP embedding space before being fed to the downstream task network.

### Step 7 — Task Training

Only the **fusion module** and **classifier head** are trained; all encoder parameters and imputed embeddings remain fixed. The fusion module uses 2-layer multi-head self-attention (4 heads, dropout, layer normalisation, mean pooling) followed by a 3-layer MLP classifier. Training minimises binary cross-entropy over all 27 genre labels. The best checkpoint (by validation F1-micro) is used for test evaluation.

---

## Missing Modality Configurations

The `--missing-config` argument controls the simulated availability pattern. Two families are supported:

### Standard: `α_image_β_text`

Each sample independently retains its image with probability $\alpha/100$ and its text with probability $\beta/100$. **One of $\alpha$ or $\beta$ must be 100.**

| Config string | Image availability | Text availability |
|---|---|---|
| `100_image_100_text` | 100 % | 100 % |
| `100_image_80_text` | 100 % | 80 % |
| `100_image_50_text` | 100 % | 50 % |
| `100_image_20_text` | 100 % | 20 % |
| `80_image_100_text` | 80 % | 100 % |
| `50_image_100_text` | 50 % | 100 % |
| `20_image_100_text` | 20 % | 100 % |

### Complex: `complex_γ_α_β`

Three disjoint groups are assigned, where $\gamma + \alpha + \beta = 100$:

| Group | Modalities present | Fraction |
|---|---|---|
| Both | image + text | $\gamma$ % |
| Image only | image | $\alpha$ % |
| Text only | text | $\beta$ % |

Example: `complex_60_20_20` → 60 % both, 20 % image-only, 20 % text-only.

Modality assignments are **deterministic** given the random seed, ensuring reproducibility across runs.

---

## Installation

```bash
pip install torch torchvision scikit-learn tqdm Pillow numpy
pip install git+https://github.com/openai/CLIP.git
```

---

## Usage

```bash
python ul4m4-mmimdb.py \
    --dataset-path /path/to/mmimdb/ \
    --missing-config 100_image_80_text \
    --k 64 \
    --kmeans-iters 50 \
    --num-epochs 15 \
    --batch-size 64 \
    --lr 5e-5 \
    --seed 42
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | `/home/office/Downloads/_Dataset/mmimdb/` | Root directory of the MM-IMDb dataset |
| `--missing-config` | `100_image_20_text` | Missing modality configuration string |
| `--k` | `64` | Number of k-means clusters |
| `--kmeans-iters` | `50` | Maximum EM iterations for partial-modality k-means |
| `--num-epochs` | `15` | Number of task training epochs |
| `--batch-size` | `64` | Batch size for embedding extraction and task training |
| `--lr` | `5e-5` | Adam learning rate |
| `--seed` | `42` | Random seed for reproducibility |

---

## Outputs

After a run completes, two files are saved in the working directory:

**`ul4m4-k{k}-e{epochs}-lr{lr}-bs{bs}-{config}.json`** — Test metrics:
```json
{
    "config": "100_image_80_text",
    "k": 64,
    "metrics": {
        "F1_Micro": 0.9123,
        "F1_Macro": 0.4891,
        "Accuracy": 0.7845
    }
}
```

**`ul4m4-k{k}-e{epochs}-lr{lr}-bs{bs}-{config}.pt`** — Full checkpoint containing the trained task network, cluster centres, modality masks, normalisation statistics, and genre list — everything required to reproduce inference from scratch.

---

## Pipeline Summary

```
Raw MM-IMDb data
      │
      ▼
MMIMDbDatasetCLIP          ← modality masking (deterministic, seeded)
      │
      ▼
Frozen CLIP ViT-B/32       ← image & text embeddings [N, 512]
      │
      ▼
Z-score normalisation      ← training stats only; applied to all splits
      │
      ▼
Stage 1: Partial k-means   ← partial-modality EM with k-means++ initialisation
      │
      ▼
Stage 2: Greedy completion ← iterative candidate selection per sample
      │
      ▼
De-normalisation           ← back to raw CLIP embedding space
      │
      ▼
FusionModule (MHSA ×2)     ← only these parameters are trained
+ Classifier (MLP ×3)
      │
      ▼
Binary cross-entropy       ← multi-label genre classification (27 classes)
```
---------------------------------------------------------------------------------------------------------------------------------
# UL4M4 on CMU-MOSI: Cluster-Guided Iterative Imputation for Missing Modalities

This repository contains the implementation of **UL4M4** applied to the **CMU-MOSI** multimodal sentiment regression benchmark. UL4M4 handles missing modalities at both training and inference time through a two-stage unsupervised approach — multi-modal clustering followed by iterative greedy completion — with no encoder fine-tuning required. For full method details, please refer to the accompanying paper.

---

## Repository Structure

```
.
├── mosi_reg.py           ← Dataset class, modality masking, collate function
├── mosi-ul4m4.py         ← Full UL4M4 pipeline (embedding → clustering → imputation → training)
└── README.md
```

---

## Dataset

### What is CMU-MOSI?

CMU-MOSI (Multimodal Opinion Sentiment Intensity) is a benchmark for **multimodal sentiment regression**. Each sample is a short video clip of a speaker expressing an opinion, and the task is to predict a continuous **sentiment score in the range [−3, +3]**, where negative values indicate negative sentiment and positive values indicate positive sentiment. Each sample contains three modalities:

- **Text** — spoken word transcripts (`.annotprocessed` files)
- **Audio** — speech waveforms at 16 kHz (`.wav` files)
- **Video** — facial video of the speaker (`.mp4` files)

### Expected Folder Structure

```
CMU-MOSI/
├── mosi_splits-70train.json          ← train / val / test split file
├── Transcript/
│   └── Segmented/
│       └── <video_id>.annotprocessed ← transcript files; each line: <segment_idx>_<text>
├── Audio/
│   └── WAV_16000/
│       └── Segmented/
│           └── <video_id>_<segment_idx>.wav
└── Video/
    └── Segmented/
        └── <video_id>_<segment_idx>.mp4
```

### Split File Format

`mosi_splits-70train.json` must contain keys `"train"`, `"val"`, and `"test"`, each mapping to a list of sample dicts:

```json
{
  "train": [
    {"name": "<video_id>_<segment_idx>", "label": 1.8},
    ...
  ],
  "val":   [...],
  "test":  [...]
}
```

The `name` field is used to locate audio and video files directly (e.g. `<name>.wav`, `<name>.mp4`) and to look up the correct line in the transcript file (lines are prefixed with `<segment_idx>_`).

---

## Data Preprocessing

### Step 1 — Dataset Loading (`mosi_reg.py`)

`MOSIDatasetRegression` reads raw text, audio, and video per sample. For each item it:

1. Parses `<video_id>.annotprocessed` to find the line starting with `<segment_idx>_` and extracts the transcript.
2. Loads `<name>.wav` using `torchaudio` (fallback: 1-second zero waveform at 16 kHz).
3. Loads `<name>.mp4` frame-by-frame using OpenCV (fallback: empty frame list).
4. Returns the continuous sentiment label as a `float32` scalar.
5. Applies **deterministic modality masking** based on the chosen missing configuration and a fixed random seed: missing text becomes `""`, missing audio becomes a zero waveform, missing video becomes an empty list.

### Step 2 — Embedding Extraction (`mosi-ul4m4.py`)

Three frozen pretrained encoders produce per-sample embeddings:

| Modality | Encoder | Embedding dim |
|---|---|---|
| Text | BERT-base-uncased (`[CLS]` token) | 768 |
| Audio | WavLM-Large (mean-pooled over time) | 1024 |
| Video | CLIP ViT-B/32 (mean-pooled over 8 sampled frames) | 768 |

Embeddings are only extracted for modalities that are present in each sample (as determined by the availability masks). Missing modalities are stored as `None` in the per-sample embedding dictionary.

For video, if a clip has fewer than 8 frames the frames are tiled to reach 8; if it has more, 8 are sampled uniformly. Frames are converted from BGR to RGB before being passed to the CLIP processor.

### Step 3 — Normalisation (training set statistics only)

Per-modality mean and standard deviation are computed **exclusively over available (non-missing) training samples** and then applied to normalise embeddings during clustering. A small epsilon (`1e-8`) is added to the standard deviation to avoid division by zero. Dev and test splits use statistics derived **solely from the training set**.

### Step 4 — Stage 1: Partial-Modality K-Means Clustering

K-means clustering is run on the normalised training embeddings using a partial-modality distance that only compares modalities present in **both** a sample and a cluster centre, normalised by the number of shared modalities and each modality's embedding dimensionality. Pairs sharing no modalities are assigned infinite distance.

Centres are initialised using **k-means++ seeding** directly on the partial-modality distance (not concatenated embeddings): the first centre is chosen uniformly at random; each subsequent centre is sampled with probability proportional to the squared partial-modality distance to the nearest already-chosen centre. Centres inherit exactly the modalities of their initialising training sample, so they can themselves be partial.

The EM loop uses **Elkan's triangle-inequality acceleration** to skip distance computations that cannot change assignments, using bounds on centre-to-centre distances. This significantly reduces the per-iteration cost for large $k$. Convergence is declared when no assignments change or all centre drifts fall below `1e-6`. Centre updates compute the mean over all cluster members that **possess** each modality; if no member possesses a modality, that modality remains absent in the centre.

### Step 5 — Stage 2: Iterative Greedy Completion

For every incomplete sample (missing at least one modality) in all three splits:

1. **Initialise $k$ candidates**: each candidate copies the sample's available modalities and fills each missing modality from the corresponding cluster centre (if the centre has it).
2. **Greedy loop**: compute partial-modality distances between all remaining candidates and all cluster centres; select the globally best (candidate, centre) pair; remove that candidate from the active set; copy any modalities it provides that the sample still lacks. Repeat until the sample is fully complete.

Dev and test splits are imputed using cluster centres derived **solely from the training set**.

### Step 6 — Task Training

Only the **fusion module** and **regression head** are trained; all encoder parameters and imputed embeddings remain fixed. The fusion module uses 1-layer multi-head self-attention (1 head, output dim 32, dropout 0.2, layer normalisation, mean pooling). The regression head is a single linear layer followed by `Hardtanh(min=-3, max=3)` to keep predictions within the valid sentiment range. Training minimises MSE loss using Adam. The best checkpoint (by validation MSE) is used for test evaluation.

> **Note on `TrainOnComplete`:** When `TrainOnComplete = True` (default), the task head is trained on the fully-observed training set (`100_text_100_audio_100_video`) rather than the missing-modality version. Imputation is still applied to val and test splits. This option is controlled by the global flag at the top of `mosi-ul4m4.py`.

---

## Missing Modality Configurations

The `--missing-config` argument (and `MOSIDatasetRegression`'s `missing_config` parameter) controls the simulated availability pattern for all three modalities. Two families are supported:

### Standard: `α_text_β_audio_γ_video`

Each sample independently retains its text with probability $\alpha/100$, audio with probability $\beta/100$, and video with probability $\gamma/100$. At least one modality should be kept at 100 % for meaningful evaluation.

| Config string | Text | Audio | Video |
|---|---|---|---|
| `100_text_100_audio_100_video` | 100 % | 100 % | 100 % |
| `20_text_100_audio_100_video` | 20 % | 100 % | 100 % |
| `100_text_20_audio_100_video` | 100 % | 20 % | 100 % |
| `100_text_100_audio_20_video` | 100 % | 100 % | 20 % |
| `100_text_20_audio_20_video` | 100 % | 20 % | 20 % |
| `20_text_100_audio_20_video` | 20 % | 100 % | 20 % |
| `20_text_20_audio_100_video` | 20 % | 20 % | 100 % |

### Complex: `complex_α_β_γ_δ_ε_ζ_η`

Seven disjoint groups cover all non-empty subsets of the three modalities, where $\alpha + \beta + \gamma + \delta + \varepsilon + \zeta + \eta = 100$:

| Parameter | Group | Modalities present |
|---|---|---|
| $\alpha$ | All three | text + audio + video |
| $\beta$ | Text only | text |
| $\gamma$ | Audio only | audio |
| $\delta$ | Video only | video |
| $\varepsilon$ | Text + Audio | text + audio |
| $\zeta$ | Text + Video | text + video |
| $\eta$ | Audio + Video | audio + video |

Example: `complex_20_20_20_10_10_10_10` → 20 % all-three, 20 % text-only, 20 % audio-only, 10 % video-only, 10 % text+audio, 10 % text+video, 10 % audio+video.

Modality assignments are **deterministic** given the random seed, ensuring reproducibility across runs.

---

## Installation

```bash
pip install torch torchvision torchaudio transformers opencv-python-headless scikit-learn scipy tqdm Pillow numpy matplotlib
```

Pretrained models are downloaded automatically on first run:
- `bert-base-uncased` (HuggingFace)
- `microsoft/wavlm-large` (HuggingFace)
- `openai/clip-vit-base-patch32` (HuggingFace)

---

## Usage

### Default run (multiple configs, 10 runs each)

```bash
python mosi-ul4m4.py
```

This runs the configurations defined in `__main__` with `k=10`, 20 epochs, lr `5e-4`, batch size 8.

### Custom configs via command line

```bash
python mosi-ul4m4.py 100_text_100_audio_100_video 20_text_100_audio_100_video complex_20_20_20_10_10_10_10
```

Each positional argument is treated as a separate missing-modality configuration. All are run with `k=70`, 20 epochs, 10 runs.

### Programmatic usage

```python
from mosi-ul4m4 import main

main(
    missing_configs=["100_text_100_audio_100_video", "20_text_100_audio_100_video"],
    num_runs=5,
    k_clusters=64,
    num_epochs=20,
    lr=5e-4,
    batch_size=8,
    audio_dir="/path/to/Audio/WAV_16000/Segmented",
    video_dir="/path/to/Video/Segmented",
    text_dir="/path/to/Transcript/Segmented",
    split_file="/path/to/mosi_splits-70train.json",
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `num_runs` | `10` | Number of repeated runs per config (results are averaged) |
| `k_clusters` | `10` / `70` | Number of k-means clusters |
| `num_epochs` | `20` | Task training epochs |
| `lr` | `5e-4` | Adam learning rate |
| `batch_size` | `8` | Batch size for embedding extraction and task training |
| `TrainOnComplete` | `True` | Train task head on fully-observed data (top of `mosi-ul4m4.py`) |
| `FuseODim` | `32` | Fusion module output dimensionality |
| `FuseLyN` | `1` | Number of MHSA layers in fusion module |
| `FuseH` | `1` | Number of attention heads |
| `DpO` | `0.2` | Dropout probability in fusion module |

---

## Outputs

Results are saved in a `results/` directory created in the working directory.

**Per-run JSON** — `cluster-k{k}-e{epochs}-lr{lr}-{config}-{run}.json`:
```json
{
    "config": "20_text_100_audio_100_video",
    "run": 1,
    "seed": 42,
    "k_clusters": 64,
    "test_metrics": {
        "MSE": 0.9821,
        "MAE": 0.7634,
        "RMSE": 0.9910,
        "Pearson_Corr": 0.7123,
        "Binary_Accuracy": 0.8012,
        "F1_Micro": 0.8012,
        "F1_Macro": 0.7891
    }
}
```

**Summary JSON** — `SUMMARY-cluster-k{k}.json`: aggregates all configs and runs into a single file.

### Reported Metrics

| Metric | Description |
|---|---|
| MSE | Mean squared error on continuous scores |
| MAE | Mean absolute error on continuous scores |
| RMSE | Root mean squared error |
| Pearson Corr | Pearson correlation between predictions and ground truth |
| Binary Accuracy | Accuracy of positive/negative sentiment classification (threshold: 0) |
| F1 Micro / Macro | F1 scores on the binarised positive/negative task |

---

## Pipeline Summary

```
Raw CMU-MOSI data
      │
      ▼
MOSIDatasetRegression      ← modality masking (deterministic, seeded)
      │
      ▼
Frozen Encoders            ← BERT (text, 768) | WavLM-Large (audio, 1024) | CLIP ViT-B/32 (video, 768)
      │
      ▼
Z-score normalisation      ← training stats only; applied during clustering
      │
      ▼
Stage 1: Partial k-means   ← k-means++ init + Elkan-accelerated EM
      │
      ▼
Stage 2: Greedy completion ← iterative candidate selection per sample
      │
      ▼
FusionModule (MHSA ×1)     ← only these parameters are trained
+ Regressor (Linear → Hardtanh[-3,3])
      │
      ▼
MSE loss                   ← continuous sentiment regression
```
--------------------------------------------------------------------------------------
# UL4M4 on Sleep-EDF: Cluster-Guided Iterative Imputation for Missing Modalities

This repository contains the implementation of **UL4M4** applied to the **Sleep-EDF** polysomnography benchmark for sleep-stage classification. UL4M4 handles missing modalities at both training and inference time through a multi-stage unsupervised approach — autoencoder pretraining, multi-modal clustering, and iterative greedy completion — with no encoder fine-tuning after pretraining. For full method details, please refer to the accompanying paper.

---

## Repository Structure

```
.
├── edf_dataset.py        ← Dataset class, missing simulation, train/val/test loaders
├── ul4m4-edf.py          ← Full UL4M4 pipeline (autoencoder → clustering → imputation → training)
└── README.md
```

---

## Standalone Usage of `edf_dataset.py`

`edf_dataset.py` can be run directly to verify the dataset loads correctly and to confirm the missing-modality simulation is working as expected (in particular, that no sample ends up with **all** modalities missing):

```bash
python edf_dataset.py
```

Expected output:
```
Train size: <N>
  x shape: (5, 250), y: <class_idx>
  client x shape: (5, 250), y: <class_idx>, has_modality: tensor([True, False, ...])
  Samples with ALL modalities missing: 0  (should be 0)
  Missing counts per modality: {0: ..., 1: ..., 2: ..., 3: ..., 4: ...}
```

The key check is that `Samples with ALL modalities missing: 0` — this confirms that the PEPSY last-lead exclusion constraint is correctly implemented.

---

## Dataset

### What is Sleep-EDF?

Sleep-EDF is a polysomnography dataset for **sleep-stage classification** across **5 classes**: Wake (W), N1, N2, N3, and REM (R). Each sample is a 30-second EEG epoch stored as a `.npy` file. The raw signal contains 7 channels; the last 2 are dropped, leaving **K = 5 channels treated as independent modalities**. Each channel is resampled/interpolated to 250 time steps and z-score standardised per channel.

### Expected Folder Structure

```
SLEEP_EDF/
└── final_data/
    ├── metadata.json             ← {sample_id: label}  (label ∈ {"W","1","2","3","4","R"})
    ├── train_metadata.json       ← [list of sample_ids for training]
    ├── test_metadata.json        ← [list of sample_ids for testing]
    └── <sample_id>.npy           ← raw signal array, shape (7, L)
```

> **Label mapping:** stages `"3"` and `"4"` are both mapped to class index 3 (N3), following standard AASM convention. There is no built-in validation split — `make_val_loader` carves a configurable fraction (default 10 %) off the training indices at runtime.

The default data root is set at the top of both files:

```python
# edf_dataset.py
EDF_DATA_ROOT = "/Documents/py-ipynb/3-EDF/data/SLEEP_EDF/final_data"

# ul4m4-edf.py
EDF_DATA_ROOT = "../data/SLEEP_EDF/final_data"
```

Change either constant to match your local path before running.

---

## Missing Modality Simulation

Missing-modality simulation follows **PEPSY's `local_missing_setup`** semantics exactly, making results directly comparable to the PEPSY paper (Table 1, EDF rows). Two parameters control the simulation:

| Parameter | Meaning |
|---|---|
| `pm` | Fraction of the K=5 modalities selected as "missing leads" for a dataset. `floor(K × pm)` leads are chosen. |
| `ps` | Fraction of samples that miss each selected lead. |

### Algorithm

1. **Select missing leads**: `floor(K × pm)` modalities are drawn at random (seeded by `seed`).
2. **Mark missing samples per lead**: for each missing lead except the last, shuffle sample indices (seeded by `seed + lead_rank`) and mark the first `ps` fraction as missing that lead.
3. **Last-lead constraint** (PEPSY fix): for the final missing lead, any sample already missing from **all** previous leads simultaneously is excluded from the candidate pool before shuffling. This prevents any sample from having every modality absent.

Missing channels are filled with `−1.0` in the signal tensor. The `has_modality` boolean tensor (shape `[K]`) indicates which channels are available for each sample: a channel is considered present if and only if `x[m, 0] != -1`.

### Configuration Examples

| Config (`pm`, `ps`) | Missing leads (K=5) | Fraction of samples missing each lead |
|---|---|---|
| `pm=0.2, ps=0.2` | 1 lead | 20 % of samples |
| `pm=0.4, ps=0.5` | 2 leads | 50 % of samples each |
| `pm=0.6, ps=0.8` | 3 leads | 80 % of samples each (default in `ul4m4-edf.py`) |

---

## Validation and Test Loaders

Because a single `(pm, ps, seed)` draw selects only **one** modality as the missing lead (when `pm = 1/K`), a naïve val/test loader would evaluate on a single missing pattern. To avoid this, `make_val_loader` and `make_test_loader` use a **comprehensive loader** strategy:

- K sub-datasets are created, one per modality as the **sole** missing lead, each covering `ps` fraction of samples.
- The K sub-datasets are concatenated into a single `ConcatDataset`.
- All sub-datasets share the same `base_seed = _EVAL_SEED = 0`, guaranteeing that val and test loaders operate under **identical missing patterns** and their accuracies are directly comparable.

```python
# Global val loader
val_loader = make_val_loader(root=EDF_DATA_ROOT, pm=0.2, ps=0.2, batch_size=128)

# Global test loader
test_loader = make_test_loader(root=EDF_DATA_ROOT, pm=0.2, ps=0.2, batch_size=128)
```

---

## Data Preprocessing (per `__getitem__`)

1. Load `<sample_id>.npy` — shape `(7, L)`.
2. Drop the last 2 channels → shape `(L, 5)`.
3. Interpolate each of the 5 channels to exactly 250 time steps.
4. Crop or zero-pad to 250 time steps and transpose → shape `(5, 250)`.
5. Z-score standardise per channel (std floored at `1e-8`).
6. Apply missing-modality masking: set `x[mod, :] = −1.0` for each missing modality.
7. Return `(x, y, has_modality)` where `has_modality[m] = (x[m, 0] != -1)`.

---

## UL4M4 Pipeline (`ul4m4-edf.py`)

### Stage 0 — Convolutional Autoencoder Pretraining (unique to Sleep-EDF)

Unlike the MM-IMDb and MOSI implementations that use large pretrained encoders (CLIP, BERT, WavLM), Sleep-EDF uses a **lightweight 1-D convolutional autoencoder trained from scratch** on raw EEG windows. This is because no off-the-shelf pretrained EEG encoder comparable to CLIP/BERT is assumed to be available.

**Architecture:**

| Component | Details |
|---|---|
| Encoder input | `(B, 1, 250)` — a single EEG channel window |
| Conv stack | 3 × [Conv1d → BatchNorm → GELU → Dropout → MaxPool1d(2)] with channels `[32, 64, 128]` and kernel size 7 |
| FC projection | Flattened features → `AE_LATENT_DIM = 128` |
| Encoder output | `(B, 128)` embedding |
| Decoder | Mirror transposed-conv stack; reconstructs `(B, 1, 250)` |

**Training:**

- Every `(sample, channel)` pair is treated as an independent training window → maximum unsupervised data usage (`N_train × 5` windows total).
- Objective: MSE reconstruction loss.
- Optimiser: Adam with cosine annealing LR schedule and gradient clipping (max norm 1.0).
- Best encoder weights (lowest reconstruction loss) are saved.
- After training the decoder is discarded; the encoder is frozen (`requires_grad = False`).

**Caching:** trained encoder weights are saved to `ae_cache/` under a filename that encodes the full architecture and training config (latent dim, channels, kernel, epochs, batch size, LR, weight decay, dropout, crop length, pm, ps). On subsequent runs with identical settings the cache is loaded directly, skipping training.

### Embedding Extraction

The frozen encoder processes each available channel of each sample independently:

- Input per channel: `(1, 1, 250)` → Output: `(128,)` embedding.
- Missing channels (marked `−1.0`) are **skipped**; they are absent from the per-sample embedding dictionary rather than stored as zeros.
- Output: `embeddings_dict = {sample_idx: {mod_idx: tensor(128)}}` with missing modalities simply not present as keys.

### Stage 1 — Partial-Modality K-Means Clustering

K-means is run on the normalised training embeddings using the partial-modality distance (only modalities present in both a sample and a centre are compared, normalised by shared modality count and embedding dimensionality). Pairs with no shared modalities are assigned infinite distance.

- **Initialisation**: k-means++ seeding directly on the partial-modality distance.
- **Acceleration**: Elkan's triangle-inequality bounds are used to skip distance computations that cannot change assignments, reducing per-iteration cost.
- **Centre update**: mean of unnormalised embeddings over cluster members that possess each modality; modalities absent from all members remain absent in the centre.
- **Convergence**: when no assignments change or all centre drifts fall below `1e-6`.

Normalisation statistics (per-modality mean and std) are computed from available training embeddings only and reused for all splits.

### Stage 2 — Iterative Greedy Completion

For every incomplete sample in all three splits (train, val, test):

1. **Build k candidates**: each candidate copies the sample's available modalities and fills each missing modality from the corresponding cluster centre (if the centre has it).
2. **Greedy loop**: find the (candidate, centre) pair with the globally smallest partial-modality distance; remove that candidate from the active set; copy any modalities it provides that the sample still lacks. Repeat until all modalities are filled.

Val and test splits are imputed using cluster centres derived **solely from the training set**.

### Stage 3 — Task Training

Only the **fusion module** and **classification head** are trained; the encoder and imputed embeddings remain fixed.

**Fusion module:** multi-head self-attention with per-modality linear projections, layer normalisation, dropout, and mean pooling over the modality dimension (following the paper's Eqs. 4–8). Configuration: `FUSE_OUT_DIM=128`, `FUSE_HEADS=8`, `FUSE_LAYERS=1`, `FUSE_DROPOUT=0.0`.

**Classification head:** `Linear(128 → 64) → ReLU → Dropout → Linear(64 → 5)`.

**Training:** SGD with class-weighted cross-entropy loss (weights `[1.0, 1.0, 0.5, 1.5, 1.0]` for W/N1/N2/N3/REM). Best checkpoint selected by validation accuracy.

---

## Installation

```bash
pip install torch torchvision torchaudio numpy scikit-learn tqdm
```

No external pretrained model downloads are required — the encoder is trained from scratch on your local data.

---

## Configuration

All hyperparameters are collected in the `CONFIG` block at the top of `ul4m4-edf.py`. Nothing else in the file needs to be changed for typical experiments.

```python
# ── paths ──────────────────────────────────────────────
EDF_DATA_ROOT = "../data/SLEEP_EDF/final_data"

# ── missing-data ────────────────────────────────────────
PM            = 0.6        # fraction of modalities that are missing leads
PS            = 0.8        # fraction of samples missing each lead

# ── Stage 0 — autoencoder ──────────────────────────────
AE_LATENT_DIM = 128        # encoder output / modality embedding dimension
AE_CHANNELS   = [32, 64, 128]
AE_KERNEL_SIZE = 7
AE_LR         = 1e-3
AE_EPOCHS     = 30
AE_BATCH_SIZE = 64
AE_DROPOUT    = 0.1
AE_WEIGHT_DECAY = 1e-4
AE_CACHE_DIR  = "ae_cache"

# ── Stage 1 — clustering ───────────────────────────────
K_CLUSTERS    = 1
MAX_KMEANS_ITERS = 100

# ── Stage 3 — fusion + head ────────────────────────────
FUSE_OUT_DIM  = 128
FUSE_HEADS    = 8
FUSE_LAYERS   = 1
FUSE_DROPOUT  = 0.0
TASK_LR       = 2e-1
TASK_EPOCHS   = 20
TASK_BATCH    = 32

# ── misc ───────────────────────────────────────────────
NUM_RUNS      = 1
BASE_SEED     = 42
```

---

## Usage

```bash
python ul4m4-edf.py
```

The script runs `NUM_RUNS` independent repetitions, each with seed `BASE_SEED + run * 100`, and prints a summary of mean ± std across runs.

---

## Outputs

Results are saved to `results/` in the working directory. Cached encoder weights are saved to `ae_cache/`.

**Results JSON** — `ul4m4_ae-pm_{PM}-ps_{PS}-k_{K}-H_{FUSE_HEADS}-L_{FUSE_LAYERS}-TLr_{TASK_LR}_.json`:

```json
{
    "config": {
        "PM": 0.6, "PS": 0.8, "K_CLUSTERS": 1,
        "AE_LATENT_DIM": 128, "AE_EPOCHS": 30,
        "TASK_EPOCHS": 20, "FUSE_OUT_DIM": 128
    },
    "runs": [
        {
            "accuracy": 0.421,
            "f1_macro": 0.1493,
            "f1_micro": 0.421,
            "f1_weighted": 0.27
        }
    ]
}
```

### Reported Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy across 5 sleep stages |
| F1 Macro | Unweighted mean F1 across all 5 classes |
| F1 Micro | Global F1 (equivalent to accuracy for single-label tasks) |
| F1 Weighted | Class-frequency-weighted mean F1 |

---

## Pipeline Summary

```
Raw Sleep-EDF .npy files
      │
      ▼
EDFBaseDataset             ← load, drop 2 channels, interpolate to 250 steps, z-score
      │
      ▼
EDFClientDataset           ← PEPSY missing simulation (pm, ps, last-lead constraint)
      │
      ▼
Stage 0: ConvAutoencoder   ← unsupervised reconstruction on raw EEG windows
         (encoder cached)     input (B,1,250) → latent (B,128) → recon (B,1,250)
      │
      ▼  [encoder frozen]
Embedding extraction       ← per-channel encoder pass; missing channels skipped
      │
      ▼
Z-score normalisation      ← training stats only; applied during clustering
      │
      ▼
Stage 1: Partial k-means   ← k-means++ init + Elkan-accelerated EM
      │
      ▼
Stage 2: Greedy completion ← iterative candidate selection per sample
      │
      ▼
FusionModule (MHSA ×1)     ← only these parameters are trained
+ Classifier (Linear → ReLU → Dropout → Linear)
      │
      ▼
Weighted cross-entropy     ← 5-class sleep-stage classification
```
