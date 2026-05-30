"""
mmimdb_ul4m4.py
===============
UL4M4 – Cluster-Guided Iterative Imputation for Missing Modalities
applied to the MM-IMDb multi-label genre classification task.

Pipeline
--------
1.  Extract frozen CLIP embeddings for the whole training set.
2.  Run Stage-1: multi-modal k-means (Elkan + k-means++ via sklearn)
    using a partial-modality distance.  Because sklearn k-means works on
    flat vectors with a single metric we implement the two-stage loop
    manually but reuse sklearn's KMeans ONLY for the initialisation of
    cluster centres (k-means++ seed selection) and delegate all distance
    / assignment / update logic to our own code so the partial-modality
    distance is respected exactly as described in the paper.
    → Actually: we initialise centres with sklearn's init='k-means++'
      (calling KMeans.fit on available-only subsets) and then run our
      own partial-distance EM iterations, matching the paper perfectly.
3.  Run Stage-2: iterative greedy completion for every incomplete
    training (and later dev / test) sample.
4.  Train `final_fusion` + `classifier` (same architecture as
    mmimdb_training.py) on the completed embeddings.
5.  Validate after every epoch; load best val checkpoint for test.
6.  Save F1-micro, F1-macro, accuracy to a JSON file.

Usage
-----
python mmimdb_ul4m4.py --missing-config 100_image_80_text \
                        --k 64 \
                        --kmeans-iters 50 \
                        --num-epochs 15 \
                        --batch-size 64 \
                        --lr 5e-5
"""

import os
import json
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import clip

from mmimdb_loader import MMIMDbDatasetCLIP, analyze_dataset, collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# Frozen encoders  (identical to mmimdb_training.py)
# ──────────────────────────────────────────────────────────────────────────────

class FrozenCLIPImageEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.device = device

    def forward(self, images):
        with torch.no_grad():
            return self.model.encode_image(images).float()   # [B, 512]


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.device = device

    def forward(self, texts):
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            return self.model.encode_text(tokens).float()    # [B, 512]


# ──────────────────────────────────────────────────────────────────────────────
# Task components  (identical to mmimdb_training.py)
# ──────────────────────────────────────────────────────────────────────────────

class FusionModule(nn.Module):
    """Multi-Head Self-Attention fusion for K modalities."""
    def __init__(self, input_dims, output_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads,
                                  batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, *features):
        projected = [proj(feat) for proj, feat in
                     zip(self.modality_projections, features)]
        seq = torch.stack(projected, dim=1)
        out = seq
        for attn, ln in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn(out, out, out)
            out = ln(out + attn_out)
        fused = out.mean(dim=1)
        return self.relu(self.output_proj(fused))


class TaskNet(nn.Module):
    """
    Thin wrapper that holds ONLY the task-relevant parameters:
      - final_fusion  (FusionModule)
      - classifier    (Sequential)
    Encoders are external; this module receives pre-computed embeddings.
    """
    def __init__(self, image_dim=512, text_dim=512,
                 fusion_dim=256, num_classes=23):
        super().__init__()
        self.final_fusion = FusionModule(
            [image_dim, text_dim],
            output_dim=fusion_dim,
            num_heads=4, num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, image_emb, text_emb):
        fused = self.final_fusion(image_emb, text_emb)
        return self.classifier(fused)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(dataset, image_enc, text_enc, device, batch_size=64):
    """
    Returns
    -------
    image_embs   : np.ndarray [N, 512]  – zeros where image was missing
    text_embs    : np.ndarray [N, 512]  – zeros where text was missing
    has_image    : np.ndarray [N]  bool
    has_text     : np.ndarray [N]  bool
    labels       : np.ndarray [N, C]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    all_img, all_txt, all_hi, all_ht, all_lbl = [], [], [], [], []

    for batch in tqdm(loader, desc="  Extracting embeddings"):
        images  = batch['images'].to(device)
        texts   = batch['texts']
        hi      = np.array(batch['has_image'], dtype=bool)
        ht      = np.array(batch['has_text'],  dtype=bool)

        img_feat = image_enc(images).cpu().numpy()   # [B, 512]
        txt_feat = text_enc(texts).cpu().numpy()     # [B, 512]

        # Zero-out features for missing modalities
        img_feat[~hi] = 0.0
        txt_feat[~ht] = 0.0

        all_img.append(img_feat)
        all_txt.append(txt_feat)
        all_hi.append(hi)
        all_ht.append(ht)
        all_lbl.append(batch['labels'].numpy())

    return (np.vstack(all_img), np.vstack(all_txt),
            np.concatenate(all_hi), np.concatenate(all_ht),
            np.vstack(all_lbl))


# ──────────────────────────────────────────────────────────────────────────────
# UL4M4  –  Stage 1: partial-modality k-means
# ──────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(image_embs, text_embs, has_image, has_text):
    """
    Compute per-modality mean and std over available samples.
    Returns (mu_img, sigma_img, mu_txt, sigma_txt) as np arrays [512].
    """
    mu_img    = image_embs[has_image].mean(axis=0)
    sigma_img = image_embs[has_image].std(axis=0) + 1e-8

    mu_txt    = text_embs[has_text].mean(axis=0)
    sigma_txt = text_embs[has_text].std(axis=0) + 1e-8

    return mu_img, sigma_img, mu_txt, sigma_txt


def normalise(embs, mu, sigma):
    return (embs - mu) / sigma


def partial_distance_matrix(
        z_img_q,  has_img_q,   # [Q, D] / [Q]  query embeddings & mask
        z_txt_q,  has_txt_q,   # [Q, D] / [Q]
        c_img,    c_has_img,   # [K, D] / [K]  centre embeddings & mask
        c_txt,    c_has_txt,   # [K, D] / [K]
        d_img=512, d_txt=512):
    """
    Compute partial-modality distance matrix D[q, j] between every
    query q and every centre j.

    Distance formula (eq. 2 in the paper):
        d(i,j) = sqrt( mean_{m in M_ij}  ||z_i^m - c_j^m||^2 / d_m )

    where M_ij is the set of modalities present in BOTH i and j.
    If M_ij is empty → distance = inf.

    Returns
    -------
    D : np.ndarray [Q, K]  (float32)
    """
    Q = z_img_q.shape[0]
    K = c_img.shape[0]

    D = np.full((Q, K), np.inf, dtype=np.float32)

    # Precompute squared differences per modality: [Q, K]
    # image: ||z_img_q[q] - c_img[j]||^2 / d_img
    # broadcast: [Q,1,D] - [1,K,D] → [Q,K,D] → mean over D → [Q,K]
    sq_img = np.sum((z_img_q[:, None, :] - c_img[None, :, :]) ** 2,
                    axis=-1) / d_img   # [Q, K]
    sq_txt = np.sum((z_txt_q[:, None, :] - c_txt[None, :, :]) ** 2,
                    axis=-1) / d_txt   # [Q, K]

    # Modality overlap mask: both query and centre have the modality
    img_overlap = has_img_q[:, None] & c_has_img[None, :]   # [Q, K]
    txt_overlap = has_txt_q[:, None] & c_has_txt[None, :]   # [Q, K]

    n_shared = img_overlap.astype(np.float32) + txt_overlap.astype(np.float32)  # [Q,K]

    # Accumulated per-modality MSE (only where overlap)
    acc = np.where(img_overlap, sq_img, 0.0) + np.where(txt_overlap, sq_txt, 0.0)

    valid = n_shared > 0
    D[valid] = np.sqrt(acc[valid] / n_shared[valid])

    return D


def kmeans_partial_modality(
        z_img, z_txt, has_img, has_txt,
        k=64, max_iter=50, seed=42,
        d_img=512, d_txt=512):
    """
    Multi-modal k-means with partial-modality distance (Stage 1 of UL4M4).

    Initialisation
    --------------
    We use sklearn KMeans with init='k-means++' and algorithm='elkan'
    to select k seed indices from samples that have AT LEAST ONE modality.
    The Elkan speed-up (triangle inequality) inside sklearn is exploited
    during the seed-selection phase; from that point we hand-roll the
    EM loop with the correct partial-modality distance.

    Parameters
    ----------
    z_img, z_txt : [N, D] normalised embeddings (zeros where missing)
    has_img, has_txt : [N] bool

    Returns
    -------
    centres_img  : [k, D]
    centres_txt  : [k, D]
    c_has_img    : [k] bool
    c_has_txt    : [k] bool
    assignments  : [N] int  cluster index for each training sample
    """
    N = z_img.shape[0]
    rng = np.random.RandomState(seed)

    # ── Initialise centres via k-means++ (sklearn does the heavy lifting) ──
    # Concatenate both modalities for samples that have both, fall back to
    # whichever is available, for the purpose of seed selection only.
    has_any  = has_img | has_txt
    feats_for_init = np.where(
        (has_img & has_txt)[:, None],
        np.hstack([z_img, z_txt]),                    # both
        np.hstack([
            np.where(has_img[:, None], z_img, z_txt),
            np.where(has_txt[:, None], z_txt, z_img)
        ])
    )  # [N, 2*D] – rough but only used for seed selection

    km_init = KMeans(
        n_clusters=k,
        init='k-means++',
        algorithm='elkan',
        max_iter=1,          # we only want the initialised centres
        n_init=1,
        random_state=seed
    )
    km_init.fit(feats_for_init)

    # Map each sklearn centre back to the nearest actual training sample
    # so centres inherit real (partial) modalities
    init_centres_flat = km_init.cluster_centers_   # [k, 2*D]
    diffs = np.sum((feats_for_init[:, None, :] -
                    init_centres_flat[None, :, :]) ** 2, axis=-1)  # [N,k]
    seed_indices = np.argmin(diffs, axis=0)        # [k]

    centres_img = z_img[seed_indices].copy()       # [k, D]
    centres_txt = z_txt[seed_indices].copy()       # [k, D]
    c_has_img   = has_img[seed_indices].copy()     # [k]
    c_has_txt   = has_txt[seed_indices].copy()     # [k]

    assignments = np.zeros(N, dtype=np.int32)

    # ── Partial-modality EM iterations ──────────────────────────────────────
    for iteration in range(max_iter):
        # --- Assignment step ---
        D_mat = partial_distance_matrix(
            z_img, has_img, z_txt, has_txt,
            centres_img, c_has_img, centres_txt, c_has_txt,
            d_img, d_txt
        )  # [N, k]
        new_assignments = np.argmin(D_mat, axis=1)

        # Check for samples with all-inf distances (shouldn't happen if
        # every sample has at least one modality, but handle gracefully)
        all_inf = np.all(np.isinf(D_mat), axis=1)
        if all_inf.any():
            new_assignments[all_inf] = rng.randint(0, k, size=all_inf.sum())

        converged = np.array_equal(new_assignments, assignments)
        assignments = new_assignments

        # --- Update step ---
        new_centres_img = np.zeros_like(centres_img)
        new_centres_txt = np.zeros_like(centres_txt)
        new_c_has_img   = np.zeros(k, dtype=bool)
        new_c_has_txt   = np.zeros(k, dtype=bool)

        for j in range(k):
            members = assignments == j
            if not members.any():
                # Empty cluster: keep old centre
                new_centres_img[j] = centres_img[j]
                new_centres_txt[j] = centres_txt[j]
                new_c_has_img[j]   = c_has_img[j]
                new_c_has_txt[j]   = c_has_txt[j]
                continue

            img_members = members & has_img
            txt_members = members & has_txt

            if img_members.any():
                new_centres_img[j] = z_img[img_members].mean(axis=0)
                new_c_has_img[j]   = True
            else:
                new_centres_img[j] = centres_img[j]   # keep old
                new_c_has_img[j]   = c_has_img[j]

            if txt_members.any():
                new_centres_txt[j] = z_txt[txt_members].mean(axis=0)
                new_c_has_txt[j]   = True
            else:
                new_centres_txt[j] = centres_txt[j]
                new_c_has_txt[j]   = c_has_txt[j]

        centres_img, centres_txt = new_centres_img, new_centres_txt
        c_has_img,   c_has_txt   = new_c_has_img,   new_c_has_txt

        if converged:
            print(f"    k-means converged at iteration {iteration+1}")
            break

    return centres_img, centres_txt, c_has_img, c_has_txt, assignments


# ──────────────────────────────────────────────────────────────────────────────
# UL4M4  –  Stage 2: iterative greedy completion
# ──────────────────────────────────────────────────────────────────────────────

def complete_sample(
        z_img_i, has_img_i,
        z_txt_i, has_txt_i,
        centres_img, c_has_img,
        centres_txt, c_has_txt,
        d_img=512, d_txt=512):
    """
    Iteratively complete ONE sample using the greedy algorithm (Stage 2).

    Returns
    -------
    out_img : np.ndarray [D]
    out_txt : np.ndarray [D]
    """
    if has_img_i and has_txt_i:
        return z_img_i.copy(), z_txt_i.copy()

    k = centres_img.shape[0]

    # Build initial candidate set: one candidate per centre
    # Each candidate = available modalities from sample + missing filled from centre
    cand_img     = np.tile(z_img_i, (k, 1))     # [k, D]
    cand_txt     = np.tile(z_txt_i, (k, 1))
    cand_has_img = np.full(k, has_img_i, dtype=bool)
    cand_has_txt = np.full(k, has_txt_i, dtype=bool)

    for f in range(k):
        if not has_img_i and c_has_img[f]:
            cand_img[f]     = centres_img[f]
            cand_has_img[f] = True
        if not has_txt_i and c_has_txt[f]:
            cand_txt[f]     = centres_txt[f]
            cand_has_txt[f] = True

    # Mutable output representation
    out_img     = z_img_i.copy()
    out_txt     = z_txt_i.copy()
    out_has_img = has_img_i
    out_has_txt = has_txt_i

    active = np.ones(k, dtype=bool)   # which candidates remain

    while not (out_has_img and out_has_txt):
        # Compute distance between all active candidates and all centres
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break

        D_mat = partial_distance_matrix(
            cand_img[active_idx],  cand_has_img[active_idx],
            cand_txt[active_idx],  cand_has_txt[active_idx],
            centres_img, c_has_img,
            centres_txt, c_has_txt,
            d_img, d_txt
        )  # [|active|, k]

        # Global minimum
        flat_idx = np.argmin(D_mat)
        ai, _ = np.unravel_index(flat_idx, D_mat.shape)
        best_cand_idx = active_idx[ai]

        # Remove from active set
        active[best_cand_idx] = False

        # Copy missing modalities from best candidate into output
        if not out_has_img and cand_has_img[best_cand_idx]:
            out_img     = cand_img[best_cand_idx].copy()
            out_has_img = True
        if not out_has_txt and cand_has_txt[best_cand_idx]:
            out_txt     = cand_txt[best_cand_idx].copy()
            out_has_txt = True

    # Fallback: if still missing (shouldn't happen with well-formed clusters)
    # fill with zero-vector
    return out_img, out_txt


def impute_split(
        z_img, z_txt, has_img, has_txt,
        centres_img, c_has_img,
        centres_txt, c_has_txt,
        d_img=512, d_txt=512):
    """
    Apply Stage-2 completion to every sample in a split.

    Returns completed_img [N,D] and completed_txt [N,D] (all present).
    """
    N = z_img.shape[0]
    out_img = z_img.copy()
    out_txt = z_txt.copy()

    incomplete = ~(has_img & has_txt)
    idx_incomplete = np.where(incomplete)[0]

    for i in tqdm(idx_incomplete, desc="  Completing samples"):
        ci, ct = complete_sample(
            z_img[i], has_img[i],
            z_txt[i], has_txt[i],
            centres_img, c_has_img,
            centres_txt, c_has_txt,
            d_img, d_txt
        )
        out_img[i] = ci
        out_txt[i] = ct

    return out_img, out_txt


# ──────────────────────────────────────────────────────────────────────────────
# Task training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

def make_tensor_loader(img_arr, txt_arr, lbl_arr, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(img_arr, dtype=torch.float32),
        torch.tensor(txt_arr, dtype=torch.float32),
        torch.tensor(lbl_arr, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_one_epoch(task_net, loader, optimizer, device):
    task_net.train()
    total_loss = 0.0
    n_batches  = 0
    for img, txt, lbl in tqdm(loader, desc="  Task train", leave=False):
        img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
        logits = task_net(img, txt)
        loss   = F.binary_cross_entropy_with_logits(logits, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate_task(task_net, loader, device, threshold=0.5):
    task_net.eval()
    all_preds, all_labels = [], []
    for img, txt, lbl in loader:
        img, txt = img.to(device), txt.to(device)
        probs  = torch.sigmoid(task_net(img, txt)).cpu().numpy()
        preds  = (probs > threshold).astype(np.float32)
        all_preds.append(preds)
        all_labels.append(lbl.numpy())
    P = np.vstack(all_preds)
    L = np.vstack(all_labels)
    f1_micro = f1_score(L.flatten(), P.flatten(), average='micro', zero_division=0)
    f1_macro = f1_score(L, P, average='macro', zero_division=0)
    acc      = accuracy_score(L.flatten(), P.flatten())
    return f1_micro, f1_macro, acc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "="*80)
    print("UL4M4 – Cluster-Guided Iterative Imputation  |  MM-IMDb")
    print("="*80)
    print(f"  missing_config : {args.missing_config}")
    print(f"  k (clusters)   : {args.k}")
    print(f"  kmeans_iters   : {args.kmeans_iters}")
    print(f"  num_epochs     : {args.num_epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  lr             : {args.lr}")
    print(f"  seed           : {args.seed}")
    print("="*80 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    dataset_path = args.dataset_path
    split_file   = os.path.join(dataset_path, "split.json")

    # ── Dataset ───────────────────────────────────────────────────────────────
    splits, genre_list, genre_to_idx, _ = analyze_dataset(
        dataset_path, split_file
    )
    num_classes = len(genre_list)

    common_kwargs = dict(
        root_dir=dataset_path,
        split_file=split_file,
        genre_list=genre_list,
        genre_to_idx=genre_to_idx,
        missing_config=args.missing_config,
        seed=args.seed
    )
    train_ds = MMIMDbDatasetCLIP(split='train', **common_kwargs)
    dev_ds   = MMIMDbDatasetCLIP(split='dev',   **common_kwargs)
    test_ds  = MMIMDbDatasetCLIP(split='test',  **common_kwargs)

    # ── Frozen encoders ───────────────────────────────────────────────────────
    print("\n[1/5] Loading frozen CLIP encoders …")
    image_enc = FrozenCLIPImageEncoder(device).to(device)
    text_enc  = FrozenCLIPTextEncoder(device).to(device)

    # ── Extract embeddings for all splits ─────────────────────────────────────
    print("\n[2/5] Extracting CLIP embeddings …")

    print("  Train split:")
    tr_img, tr_txt, tr_hi, tr_ht, tr_lbl = extract_embeddings(
        train_ds, image_enc, text_enc, device, batch_size=args.batch_size)
    print("  Dev split:")
    dv_img, dv_txt, dv_hi, dv_ht, dv_lbl = extract_embeddings(
        dev_ds,   image_enc, text_enc, device, batch_size=args.batch_size)
    print("  Test split:")
    te_img, te_txt, te_hi, te_ht, te_lbl = extract_embeddings(
        test_ds,  image_enc, text_enc, device, batch_size=args.batch_size)

    # Free GPU memory used by encoders
    del image_enc, text_enc
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── Stage 1: Compute normalisation statistics (training set only) ─────────
    print("\n[3/5] UL4M4 Stage-1 – multi-modal k-means …")
    mu_img, sigma_img, mu_txt, sigma_txt = compute_norm_stats(
        tr_img, tr_txt, tr_hi, tr_ht
    )

    # Normalise all splits (zeros for missing entries stay zero;
    # they are masked out by has_img / has_txt during distance computation)
    ntr_img = np.where(tr_hi[:, None], normalise(tr_img, mu_img, sigma_img), 0.0)
    ntr_txt = np.where(tr_ht[:, None], normalise(tr_txt, mu_txt, sigma_txt), 0.0)

    ndv_img = np.where(dv_hi[:, None], normalise(dv_img, mu_img, sigma_img), 0.0)
    ndv_txt = np.where(dv_ht[:, None], normalise(dv_txt, mu_txt, sigma_txt), 0.0)

    nte_img = np.where(te_hi[:, None], normalise(te_img, mu_img, sigma_img), 0.0)
    nte_txt = np.where(te_ht[:, None], normalise(te_txt, mu_txt, sigma_txt), 0.0)

    centres_img, centres_txt, c_has_img, c_has_txt, _ = kmeans_partial_modality(
        ntr_img, ntr_txt, tr_hi, tr_ht,
        k=args.k,
        max_iter=args.kmeans_iters,
        seed=args.seed
    )
    print(f"  Centres with image: {c_has_img.sum()} / {args.k}")
    print(f"  Centres with text : {c_has_txt.sum()} / {args.k}")

    # ── Stage 2: Complete missing modalities ──────────────────────────────────
    print("\n[4/5] UL4M4 Stage-2 – iterative greedy completion …")

    print("  Train:")
    ctr_img, ctr_txt = impute_split(
        ntr_img, ntr_txt, tr_hi, tr_ht,
        centres_img, c_has_img, centres_txt, c_has_txt)
    print("  Dev:")
    cdv_img, cdv_txt = impute_split(
        ndv_img, ndv_txt, dv_hi, dv_ht,
        centres_img, c_has_img, centres_txt, c_has_txt)
    print("  Test:")
    cte_img, cte_txt = impute_split(
        nte_img, nte_txt, te_hi, te_ht,
        centres_img, c_has_img, centres_txt, c_has_txt)

    # ── De-normalise completed embeddings back to CLIP embedding space ───────
    # UL4M4 operates internally in z-score space (clustering, distances,
    # centre updates all use normalised features).  The downstream task net
    # (final_fusion + classifier) was designed for raw CLIP embeddings, so
    # we invert the z-score transform: x = z * sigma + mu.
    print("\n  De-normalising completed embeddings → raw CLIP space …")
    ctr_img = ctr_img * sigma_img + mu_img
    ctr_txt = ctr_txt * sigma_txt + mu_txt

    cdv_img = cdv_img * sigma_img + mu_img
    cdv_txt = cdv_txt * sigma_txt + mu_txt

    cte_img = cte_img * sigma_img + mu_img
    cte_txt = cte_txt * sigma_txt + mu_txt

    # ── Task training ─────────────────────────────────────────────────────────
    print("\n[5/5] Task training (final_fusion + classifier) …")

    train_loader = make_tensor_loader(ctr_img, ctr_txt, tr_lbl,
                                      args.batch_size, shuffle=True)
    dev_loader   = make_tensor_loader(cdv_img, cdv_txt, dv_lbl,
                                      args.batch_size, shuffle=False)
    test_loader  = make_tensor_loader(cte_img, cte_txt, te_lbl,
                                      args.batch_size, shuffle=False)

    # Embedding dimensionality after de-normalisation is still 512
    task_net = TaskNet(
        image_dim=512, text_dim=512,
        fusion_dim=256, num_classes=num_classes
    ).to(device)

    n_params = sum(p.numel() for p in task_net.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(task_net.parameters(), lr=args.lr)

    best_val_micro  = -1.0
    best_state      = None

    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(task_net, train_loader, optimizer, device)
        val_micro, val_macro, val_acc = evaluate_task(task_net, dev_loader, device)
        marker = ""
        if val_micro > best_val_micro:
            best_val_micro = val_micro
            best_state     = copy.deepcopy(task_net.state_dict())
            marker = "  ★ best"
        print(f"  Epoch {epoch+1:>3}/{args.num_epochs} | "
              f"loss {train_loss:.4f} | "
              f"val F1-micro {val_micro:.4f} | "
              f"val F1-macro {val_macro:.4f} | "
              f"val acc {val_acc:.4f}"
              f"{marker}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n── Test evaluation (best val checkpoint) ──")
    task_net.load_state_dict(best_state)
    test_micro, test_macro, test_acc = evaluate_task(task_net, test_loader, device)
    print(f"  F1-micro : {test_micro:.4f}")
    print(f"  F1-macro : {test_macro:.4f}")
    print(f"  Accuracy : {test_acc:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config"        : args.missing_config,
        "k"             : args.k,
        "kmeans_iters"  : args.kmeans_iters,
        "num_epochs"    : args.num_epochs,
        "batch_size"    : args.batch_size,
        "lr"            : args.lr,
        "seed"          : args.seed,
        "metrics": {
            "F1_Micro" : float(test_micro),
            "F1_Macro" : float(test_macro),
            "Accuracy" : float(test_acc)
        }
    }
    out_name = (f"ul4m4-k{args.k}"
                f"-e{args.num_epochs}"
                f"-lr{args.lr}"
                f"-bs{args.batch_size}"
                f"-{args.missing_config}.json")
    with open(out_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to {out_name}")

    # Optionally save model
    torch.save({
        'task_net_state_dict' : task_net.state_dict(),
        'centres_img'         : centres_img,
        'centres_txt'         : centres_txt,
        'c_has_img'           : c_has_img,
        'c_has_txt'           : c_has_txt,
        'mu_img'              : mu_img,
        'sigma_img'           : sigma_img,
        'mu_txt'              : mu_txt,
        'sigma_txt'           : sigma_txt,
        'genre_list'          : genre_list,
        'config'              : vars(args)
    }, out_name.replace('.json', '.pt'))
    print(f"✓ Model checkpoint saved to {out_name.replace('.json', '.pt')}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UL4M4 – Cluster-Guided Iterative Imputation on MM-IMDb"
    )
    parser.add_argument("--dataset-path",    type=str,
                        default="/home/office/Downloads/_Dataset/mmimdb/",
                        help="Root path of the MM-IMDb dataset")
    parser.add_argument("--missing-config",  type=str,
                        default="100_image_20_text",
                        help="Missing modality config (same format as mmimdb_loader.py)")
    parser.add_argument("--k",               type=int,   default=64,
                        help="Number of k-means clusters")
    parser.add_argument("--kmeans-iters",    type=int,   default=50,
                        help="Max EM iterations for partial-modality k-means")
    parser.add_argument("--num-epochs",      type=int,   default=15,
                        help="Number of task-training epochs")
    parser.add_argument("--batch-size",      type=int,   default=64,
                        help="Batch size (both embedding extraction and task training)")
    parser.add_argument("--lr",              type=float, default=5e-5,
                        help="Adam learning rate")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed")
    args = parser.parse_args()
    main(args)