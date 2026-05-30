"""
UL4M4 for Sleep-EDF  —  with Trainable Convolutional Autoencoder
=================================================================

Replaces the frozen BENDR encoder with a lightweight 1-D convolutional
autoencoder that is trained from scratch.

Pipeline
--------
  Stage 0  (NEW)  –  Unsupervised autoencoder pretraining on raw EEG windows.
                      The encoder half is kept; the decoder is discarded.
  Stage 1          –  Multi-modal k-means clustering on frozen encoder embeddings
                      (partial-modality distance, Elkan variant).
  Stage 2          –  Iterative greedy imputation of missing modality embeddings.
  Stage 3          –  Supervised training of the fusion + classification head
                      on the completed (imputed) embeddings.

All tuneable knobs are collected in the CONFIG block below — nothing else
in the file needs to be touched for typical experiments.
"""

# ============================================================
#  CONFIG  —  every hyper-parameter lives here
# ============================================================

# ── paths ──────────────────────────────────────────────────
EDF_DATA_ROOT = "../data/SLEEP_EDF/final_data"

# ── data ───────────────────────────────────────────────────
N_MODALITIES  = 5          # EEG channels treated as separate modalities
N_CLASSES     = 5          # W, N1, N2, N3, REM
CROP_LENGTH   = 250        # time-steps per epoch after interpolation
CLASS_WEIGHTS = [1.0, 1.0, 0.5, 1.5, 1.0]

# ── federated / missing-data ────────────────────────────────
NUM_CLIENTS   = 32
PM            = 0.6        # fraction of modalities that are missing leads
PS            = 0.8        # fraction of samples missing each lead
IID           = True
ALPHA         = 0.5        # Dirichlet α (only used when IID=False)
VAL_FRACTION  = 0.1

# ── Stage 0 — autoencoder pretraining ──────────────────────
AE_LATENT_DIM      = 128   # encoder output dimension (= modality embedding dim)
AE_CHANNELS        = [32, 64, 128]   # conv channel progression (encoder)
AE_KERNEL_SIZE     = 7
AE_LR              = 1e-3
AE_EPOCHS          = 30
AE_BATCH_SIZE      = 64
AE_UNSUP_EPOCHS    = 10    # within a mixed run: how many epochs stay unsupervised
                            # (set to AE_EPOCHS to train fully unsupervised first)
AE_DROPOUT         = 0.1
AE_WEIGHT_DECAY    = 1e-4
AE_CACHE_DIR       = "ae_cache"  # directory where trained encoder weights are saved

# ── Stage 1 — clustering ───────────────────────────────────
K_CLUSTERS        = 1
MAX_KMEANS_ITERS  = 100

# ── Stage 3 — fusion + head ────────────────────────────────
FUSE_OUT_DIM  = 128
FUSE_HEADS    = 8
FUSE_LAYERS   = 1
FUSE_DROPOUT  = 0.0
TASK_LR       = 2e-1
TASK_EPOCHS   = 20
TASK_BATCH    = 32

# ── misc ───────────────────────────────────────────────────
DEVICE        = "cuda"
NUM_RUNS      = 1
BASE_SEED     = 42

# ============================================================
#  IMPORTS
# ============================================================

import os, copy, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  DATASET  —  imported directly from edf_dataset.py
# ============================================================
from edf_dataset import (
    EDFBaseDataset, EDFClientDataset,
    make_val_loader, make_test_loader,
    LABEL2IDX, _EVAL_SEED,
)


# ============================================================
#  STAGE 0 — CONVOLUTIONAL AUTOENCODER
# ============================================================

class ConvEncoder(nn.Module):
    """
    1-D convolutional encoder.

    Input : (B, 1, CROP_LENGTH)   — a single EEG channel window
    Output: (B, AE_LATENT_DIM)    — fixed-size embedding
    """
    def __init__(self,
                 in_len:     int   = CROP_LENGTH,
                 channels:   list  = AE_CHANNELS,
                 kernel_size: int  = AE_KERNEL_SIZE,
                 latent_dim: int   = AE_LATENT_DIM,
                 dropout:    float = AE_DROPOUT):
        super().__init__()
        layers = []
        in_ch  = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # Compute flattened size after pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_len)
            flat  = self.conv_stack(dummy).flatten(1).shape[1]

        self.fc = nn.Linear(flat, latent_dim)

    def forward(self, x):                          # x: (B, 1, L)
        h = self.conv_stack(x).flatten(1)
        return self.fc(h)                           # (B, latent_dim)


class ConvDecoder(nn.Module):
    """
    Mirror decoder that reconstructs the raw signal from a latent vector.
    Only used during Stage 0; discarded afterwards.
    """
    def __init__(self,
                 out_len:    int   = CROP_LENGTH,
                 channels:   list  = AE_CHANNELS,
                 kernel_size: int  = AE_KERNEL_SIZE,
                 latent_dim: int   = AE_LATENT_DIM):
        super().__init__()
        # Number of pooling layers = len(channels)
        n_pools    = len(channels)
        reduced_len = out_len // (2 ** n_pools)
        self.reduced_len = reduced_len
        self.first_ch    = channels[-1]
        self.fc = nn.Linear(latent_dim, self.first_ch * reduced_len)

        layers = []
        rev_ch = list(reversed(channels))
        for i, out_ch in enumerate(rev_ch[1:] + [1]):
            layers += [
                nn.ConvTranspose1d(rev_ch[i], out_ch, kernel_size * 2,
                                   stride=2, padding=kernel_size // 2,
                                   output_padding=0, bias=False),
                nn.BatchNorm1d(out_ch) if out_ch > 1 else nn.Identity(),
                nn.GELU() if out_ch > 1 else nn.Identity(),
            ]
        self.deconv_stack = nn.Sequential(*layers)
        self.out_len = out_len

    def forward(self, z):                          # z: (B, latent_dim)
        h = self.fc(z).view(-1, self.first_ch, self.reduced_len)
        x = self.deconv_stack(h)
        # Trim / pad to exact output length
        if x.shape[-1] > self.out_len:
            x = x[..., :self.out_len]
        elif x.shape[-1] < self.out_len:
            x = F.pad(x, (0, self.out_len - x.shape[-1]))
        return x                                   # (B, 1, out_len)


class EEGAutoencoder(nn.Module):
    """Full autoencoder (encoder + decoder)."""
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


def _ae_cache_path() -> str:
    """
    Build a filename that uniquely identifies the current AE architecture,
    training config, and missing-data regime (pm, ps).  Any change to any
    of these produces a different filename, so the old cache is never
    silently reused.

    Note: the AE itself trains on fully complete data, but pm/ps define the
    experimental regime and different configs should have independent caches.
    """
    ch_str = "-".join(str(c) for c in AE_CHANNELS)
    name   = (f"encoder"
              f"_lat{AE_LATENT_DIM}"
              f"_ch{ch_str}"
              f"_k{AE_KERNEL_SIZE}"
              f"_ep{AE_EPOCHS}"
              f"_bs{AE_BATCH_SIZE}"
              f"_lr{AE_LR}"
              f"_wd{AE_WEIGHT_DECAY}"
              f"_do{AE_DROPOUT}"
              f"_crop{CROP_LENGTH}"
              f"_pm{PM}_ps{PS}"
              ".pt")
    return os.path.join(AE_CACHE_DIR, name)


def pretrain_autoencoder(train_base: EDFBaseDataset,
                         train_indices: list,
                         device: str = DEVICE) -> ConvEncoder:
    """
    Stage 0: train the autoencoder on raw EEG windows (reconstruction only).

    If a cached encoder with the exact same architecture + training config
    already exists on disk, it is loaded directly and training is skipped.
    Otherwise the autoencoder is trained and the encoder weights are saved
    to disk for future runs.

    Each sample has N_MODALITIES channels; every available channel is treated
    as an independent training window → maximum unsupervised data usage.

    Parameters
    ----------
    train_base    : EDFBaseDataset (all channels available, no masking)
    train_indices : list of sample indices to use for pretraining
    device        : torch device string

    Returns
    -------
    Frozen ConvEncoder (requires_grad = False everywhere)
    """
    cache_path = _ae_cache_path()

    print(f"\n{'='*70}")
    print("STAGE 0 — Autoencoder Pretraining (unsupervised)")
    print(f"  latent_dim={AE_LATENT_DIM}  channels={AE_CHANNELS}  "
          f"kernel={AE_KERNEL_SIZE}  epochs={AE_EPOCHS}")
    print(f"  cache: {cache_path}")
    print(f"{'='*70}")

    encoder = ConvEncoder().to(device)

    # ── Try to load from cache ─────────────────────────────
    if os.path.exists(cache_path):
        print(f"✓ Found cached encoder — loading weights, skipping training.")
        encoder.load_state_dict(torch.load(cache_path, map_location=device))
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        print(f"{'='*70}\n")
        return encoder

    # ── Cache miss: train from scratch ─────────────────────
    print("  No cache found — training from scratch …")
    ae  = EEGAutoencoder().to(device)
    opt = torch.optim.Adam(ae.parameters(),
                           lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)

    class _RawChannelDataset(Dataset):
        """Returns every (sample, channel) pair as a 1-D window."""
        def __init__(self, base, indices):
            self.base, self.indices = base, indices
        def __len__(self):
            return len(self.indices) * N_MODALITIES
        def __getitem__(self, idx):
            si  = idx // N_MODALITIES
            mod = idx  % N_MODALITIES
            x, _ = self.base[self.indices[si]]
            return torch.tensor(x[mod], dtype=torch.float32).unsqueeze(0)

    raw_loader = DataLoader(
        _RawChannelDataset(train_base, train_indices),
        batch_size=AE_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    best_loss, best_enc_w = float('inf'), None

    for epoch in range(1, AE_EPOCHS + 1):
        ae.train()
        epoch_loss, n_batches = 0.0, 0
        for xw in raw_loader:
            xw = xw.to(device)
            xhat, _ = ae(xw)
            loss = F.mse_loss(xhat, xw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); n_batches += 1
        sched.step()
        avg = epoch_loss / max(n_batches, 1)
        if avg < best_loss:
            best_loss  = avg
            best_enc_w = copy.deepcopy(ae.encoder.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{AE_EPOCHS}  recon_loss={avg:.5f}"
                  f"{'  ← best' if avg == best_loss else ''}")

    # Restore best weights, freeze, and save to cache
    ae.encoder.load_state_dict(best_enc_w)
    for p in ae.encoder.parameters():
        p.requires_grad = False
    ae.encoder.eval()

    os.makedirs(AE_CACHE_DIR, exist_ok=True)
    torch.save(ae.encoder.state_dict(), cache_path)
    print(f"\n✓ Pretraining done  |  best recon loss: {best_loss:.5f}")
    print(f"✓ Encoder weights saved → {cache_path}")
    print(f"{'='*70}\n")
    return ae.encoder


# ============================================================
#  EMBEDDING EXTRACTION
# ============================================================

def extract_embeddings(encoder: ConvEncoder,
                       client_ds: EDFClientDataset,
                       device: str = DEVICE) -> tuple:
    """
    Run the frozen encoder over every sample×channel, respecting missing masks.

    Returns
    -------
    embeddings_dict : {local_idx: {mod_idx: tensor(AE_LATENT_DIM)}}
                      missing channels are absent from the inner dict
    labels          : list of int
    """
    loader = DataLoader(client_ds, batch_size=AE_BATCH_SIZE,
                        shuffle=False, num_workers=0)
    encoder.eval()
    embeddings_dict: dict = {}
    labels: list         = []
    gi = 0

    with torch.no_grad():
        for xb, yb, has in tqdm(loader, desc="Extracting embeddings", leave=False):
            # xb: (B, N_MODALITIES, CROP_LENGTH)
            B = xb.shape[0]
            for bi in range(B):
                emb_sample = {}
                for mod in range(N_MODALITIES):
                    if has[bi, mod].item():
                        win = xb[bi, mod].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
                        z   = encoder(win).squeeze(0).cpu()                      # (D,)
                        emb_sample[mod] = z
                embeddings_dict[gi] = emb_sample
                labels.append(yb[bi].item())
                gi += 1

    return embeddings_dict, labels


# ============================================================
#  STAGE 1 — CLUSTER-GUIDED IMPUTATION  (UL4M4 — paper faithful)
# ============================================================

class ClusterGuidedImputation:
    """
    Implements the UL4M4 two-stage imputation from the paper:
      Stage 1 — partial-modality k-means (Elkan variant, k-means++ init)
      Stage 2 — iterative greedy completion
    """
    def __init__(self, modality_dim: int = AE_LATENT_DIM,
                 k_clusters: int = K_CLUSTERS,
                 max_iters:  int = MAX_KMEANS_ITERS,
                 seed: int = BASE_SEED):
        self.D          = modality_dim
        self.k          = k_clusters
        self.max_iters  = max_iters
        self.seed       = seed
        self.mod_names  = list(range(N_MODALITIES))
        self.mod_dim    = {m: modality_dim for m in self.mod_names}
        self.norm_stats = None      # {mod: (mean, std)}
        self.centres    = None      # list of k dicts {mod: tensor}
        np.random.seed(seed); torch.manual_seed(seed)

    # ── helpers ────────────────────────────────────────────

    def _compute_norm_stats(self, emb_dict):
        stats = {}
        for mod in self.mod_names:
            vecs = [emb_dict[i][mod] for i in emb_dict if mod in emb_dict[i]]
            if vecs:
                stacked = torch.stack(vecs)
                mu  = stacked.mean(0)
                sig = stacked.std(0)
                sig = torch.where(sig > 1e-8, sig, torch.ones_like(sig))
            else:
                mu  = torch.zeros(self.D)
                sig = torch.ones(self.D)
            stats[mod] = (mu, sig)
        return stats

    def _norm(self, z, mod):
        mu, sig = self.norm_stats[mod]
        return (z - mu) / sig

    def _partial_dist(self, a: dict, b: dict, normed=True) -> float:
        """Eq. 2 from the paper."""
        shared = [m for m in self.mod_names if m in a and m in b]
        if not shared:
            return float('inf')
        total = 0.0
        for m in shared:
            ea = a[m] if normed else self._norm(a[m], m)
            eb = b[m] if normed else self._norm(b[m], m)
            total += ((ea - eb) ** 2).sum().item() / self.mod_dim[m]
        return (total / len(shared)) ** 0.5

    # ── Stage 1a: k-means++ init ───────────────────────────

    def _init_centres_pp(self, norm_samples):
        ids = list(norm_samples.keys())
        N   = len(ids)
        chosen = [ids[np.random.randint(N)]]
        d2 = np.full(N, np.inf)

        def _update(cid):
            for ni, sid in enumerate(ids):
                d = self._partial_dist(norm_samples[sid], norm_samples[cid])
                if d == float('inf'): d = 1e9
                d2[ni] = min(d2[ni], d * d)

        _update(chosen[0])
        for _ in range(self.k - 1):
            tot = d2.sum()
            probs = d2 / tot if tot > 0 else np.ones(N) / N
            ni  = np.random.choice(N, p=probs)
            chosen.append(ids[ni])
            _update(chosen[-1])
        return chosen

    # ── Stage 1b: Elkan k-means ────────────────────────────

    def fit(self, emb_dict: dict):
        print(f"\n{'='*70}")
        print(f"STAGE 1 — Clustering  (k={self.k}, max_iters={self.max_iters})")
        print(f"{'='*70}")

        self.norm_stats = self._compute_norm_stats(emb_dict)
        ids = list(emb_dict.keys())
        N, K = len(ids), self.k

        # Pre-normalise
        ns = {sid: {m: self._norm(emb_dict[sid][m], m)
                    for m in emb_dict[sid]}
              for sid in ids}

        # Init
        seed_ids = self._init_centres_pp(ns)

        def _norm_c(raw_cs):
            return [{m: self._norm(c[m], m) for m in c} for c in raw_cs]

        centres    = [{m: emb_dict[sid][m].clone() for m in emb_dict[sid]}
                      for sid in seed_ids]
        nc         = _norm_c(centres)

        lower  = np.zeros((N, K))
        upper  = np.full(N, np.inf)
        asgn   = np.zeros(N, dtype=int)
        r      = np.ones(N, dtype=bool)

        # Initial exact distances
        for ni, sid in enumerate(ids):
            bd, bc = np.inf, 0
            for ci in range(K):
                d = self._partial_dist(ns[sid], nc[ci])
                lower[ni, ci] = d
                if d < bd: bd, bc = d, ci
            asgn[ni], upper[ni] = bc, bd

        def _cc_matrix(ncs):
            D = np.zeros((K, K))
            for a in range(K):
                for b in range(a + 1, K):
                    d = self._partial_dist(ncs[a], ncs[b])
                    D[a, b] = D[b, a] = d
            return D

        for it in range(self.max_iters):
            cc   = _cc_matrix(nc)
            s    = 0.5 * np.nanmin(
                np.where(np.eye(K, dtype=bool), np.inf, cc), axis=1)

            changed = False
            for ni, sid in enumerate(ids):
                cc_cur = asgn[ni]
                if upper[ni] <= s[cc_cur]:
                    continue
                for ci in range(K):
                    if ci == cc_cur: continue
                    if upper[ni] <= lower[ni, ci]: continue
                    if upper[ni] <= 0.5 * cc[cc_cur, ci]: continue
                    if r[ni]:
                        d = self._partial_dist(ns[sid], nc[cc_cur])
                        upper[ni] = d; lower[ni, cc_cur] = d; r[ni] = False
                    if upper[ni] <= lower[ni, ci]: continue
                    d2 = self._partial_dist(ns[sid], nc[ci])
                    lower[ni, ci] = d2
                    if d2 < upper[ni]:
                        asgn[ni] = ci; upper[ni] = d2; cc_cur = ci; changed = True

            # Update centres (unnormalised means — Eq. 3)
            new_centres = []
            for ci in range(K):
                members = [ids[ni] for ni, a in enumerate(asgn) if a == ci]
                c = {}
                for mod in self.mod_names:
                    vecs = [emb_dict[m_id][mod]
                            for m_id in members if mod in emb_dict[m_id]]
                    if vecs:
                        c[mod] = torch.stack(vecs).mean(0)
                new_centres.append(c)

            new_nc = _norm_c(new_centres)
            delta  = np.array([self._partial_dist(nc[ci], new_nc[ci])
                                for ci in range(K)])

            for ni in range(N):
                lower[ni] = np.maximum(0, lower[ni] - delta)
                upper[ni] += delta[asgn[ni]]
                r[ni]      = True

            nc, centres = new_nc, new_centres
            if (it + 1) % 10 == 0:
                print(f"  iter {it+1:>3d}  max_drift={delta.max():.2e}"
                      f"  changed={changed}")
            if not changed or np.all(delta < 1e-6):
                print(f"  Converged at iter {it+1}"); break

        self.centres = centres
        final_asgn   = {sid: int(asgn[ni]) for ni, sid in enumerate(ids)}
        counts = np.bincount(asgn, minlength=K)
        print(f"\n  Cluster sizes: {counts.tolist()}")
        print(f"{'='*70}\n")
        return final_asgn

    # ── Stage 2: greedy completion (Algorithm 2) ───────────

    def impute(self, sample: dict) -> dict:
        """Impute all missing modalities for one sample."""
        missing = [m for m in self.mod_names if m not in sample]
        if not missing:
            return sample

        # Build initial candidates
        cands = []
        for c in self.centres:
            cand = dict(sample)
            for m in missing:
                if m in c:
                    cand[m] = c[m].clone()
            cands.append(cand)

        result  = dict(sample)
        avail   = set(range(len(cands)))
        missing = list(missing)

        while missing and avail:
            best_d, best_ci = np.inf, None
            for ci in avail:
                for c in self.centres:
                    d = self._partial_dist(cands[ci], c, normed=False)
                    if d < best_d:
                        best_d, best_ci = d, ci
            if best_ci is None: break
            bc = cands[best_ci]; avail.remove(best_ci)
            for m in missing[:]:
                if m in bc:
                    result[m] = bc[m].clone()
                    missing.remove(m)

        return result


# ============================================================
#  FUSION MODULE  (Eqs. 4–8 of the paper)
# ============================================================

class FusionModule(nn.Module):
    def __init__(self, n_mods: int = N_MODALITIES,
                 in_dim: int = AE_LATENT_DIM,
                 out_dim: int = FUSE_OUT_DIM,
                 n_heads: int = FUSE_HEADS,
                 n_layers: int = FUSE_LAYERS,
                 dropout: float = FUSE_DROPOUT):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(in_dim, out_dim)
                                    for _ in range(n_mods)])
        self.attn  = nn.ModuleList([
            nn.MultiheadAttention(out_dim, n_heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim)
                                    for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)
        self.out   = nn.Linear(out_dim, out_dim)

    def forward(self, *feats):
        # feats: each (B, in_dim)
        seq = torch.stack([p(f) for p, f in zip(self.projs, feats)], dim=1)
        for attn, norm in zip(self.attn, self.norms):
            a, _ = attn(seq, seq, seq)
            seq  = norm(seq + self.drop(a))
        fused = seq.mean(dim=1)
        return F.relu(self.out(fused))


class UL4M4Model(nn.Module):
    """Combines frozen encoder + imputer + fusion + task head."""
    def __init__(self, encoder: ConvEncoder):
        super().__init__()
        self.encoder = encoder          # frozen after Stage 0
        self.fusion  = FusionModule()
        self.head    = nn.Sequential(
            nn.Linear(FUSE_OUT_DIM, 64),
            nn.ReLU(),
            nn.Dropout(FUSE_DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, *feats):          # each feat: (B, AE_LATENT_DIM)
        return self.head(self.fusion(*feats))


# ============================================================
#  STAGE 3 — TASK TRAINING
# ============================================================

def train_task(model: UL4M4Model,
               complete_train: dict, train_labels: list,
               complete_val:   dict, val_labels:   list,
               device: str = DEVICE) -> UL4M4Model:
    """Train fusion + head; encoder stays frozen."""
    print(f"\n{'='*70}")
    print("STAGE 3 — Task Training (fusion + head)")
    print(f"{'='*70}")

    n_tr, n_va = len(complete_train), len(complete_val)

    def _stack(d, labels):
        mods = [torch.stack([d[i][m] for i in range(len(d))]).to(device)
                for m in range(N_MODALITIES)]
        lbl  = torch.tensor(labels, dtype=torch.long).to(device)
        return mods, lbl

    tr_mods, tr_lbl = _stack(complete_train, train_labels)
    va_mods, va_lbl = _stack(complete_val,   val_labels)

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    crit    = nn.CrossEntropyLoss(weight=weights)
    opt     = torch.optim.SGD(
        list(model.fusion.parameters()) + list(model.head.parameters()),
        lr=TASK_LR)

    best_acc, best_w = 0.0, None

    for ep in range(1, TASK_EPOCHS + 1):
        model.train(); model.encoder.eval()
        idx = torch.randperm(n_tr)
        ep_loss = 0.0; nb = 0

        for s in range(0, n_tr, TASK_BATCH):
            bi = idx[s:s + TASK_BATCH]
            bm = [m[bi] for m in tr_mods]
            bl = tr_lbl[bi]
            opt.zero_grad()
            logits = model(*bm)
            loss   = crit(logits, bl)
            loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            va_log = model(*va_mods)
            preds  = va_log.argmax(1).cpu().numpy()
            acc    = accuracy_score(va_lbl.cpu().numpy(), preds)

        if acc > best_acc:
            best_acc = acc
            best_w   = copy.deepcopy(model.state_dict())
            print(f"  ★ ep {ep:>3d}  loss={ep_loss/nb:.4f}  val_acc={acc:.4f}")
        elif ep % 10 == 0:
            print(f"    ep {ep:>3d}  loss={ep_loss/nb:.4f}  val_acc={acc:.4f}")

    model.load_state_dict(best_w)
    print(f"\n✓ Best val acc: {best_acc:.4f}")
    print(f"{'='*70}\n")
    return model


# ============================================================
#  EVALUATION
# ============================================================

def evaluate(model: UL4M4Model, complete_test: dict,
             test_labels: list, device: str = DEVICE) -> dict:
    model.eval()
    n = len(complete_test)
    mods = [torch.stack([complete_test[i][m] for i in range(n)]).to(device)
            for m in range(N_MODALITIES)]
    lbl_np = np.array(test_labels)
    with torch.no_grad():
        preds = model(*mods).argmax(1).cpu().numpy()

    return {
        "accuracy":  float(accuracy_score(lbl_np, preds)),
        "f1_macro":  float(f1_score(lbl_np, preds, average='macro',
                                    zero_division=0)),
        "f1_micro":  float(f1_score(lbl_np, preds, average='micro',
                                    zero_division=0)),
        "f1_weighted": float(f1_score(lbl_np, preds, average='weighted',
                                      zero_division=0)),
    }


# ============================================================
#  MAIN PIPELINE
# ============================================================

def run_once(seed: int, root: str = EDF_DATA_ROOT,
             device: str = DEVICE) -> dict:
    """One complete run: Stage 0 → 1 → 2 → 3 → eval."""

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # ── Load data ──────────────────────────────────────────
    train_base = EDFBaseDataset(root, train=True)
    all_tr_idx = list(range(len(train_base)))
    random.Random(seed).shuffle(all_tr_idx)
    vs         = max(1, int(len(all_tr_idx) * VAL_FRACTION))
    val_idx    = all_tr_idx[:vs]
    train_idx  = all_tr_idx[vs:]

    train_cl = EDFClientDataset(train_base, train_idx, PM, PS, seed=seed)
    val_cl   = EDFClientDataset(train_base, val_idx,   PM, PS, seed=seed + 10000)

    test_base = EDFBaseDataset(root, train=False)
    test_cl   = EDFClientDataset(test_base, list(range(len(test_base))),
                                 PM, PS, seed=_EVAL_SEED)

    # ── Stage 0: pretrain autoencoder ──────────────────────
    encoder = pretrain_autoencoder(train_base, train_idx, device)

    # ── Extract embeddings ─────────────────────────────────
    print("Extracting embeddings …")
    tr_emb,  tr_lbl  = extract_embeddings(encoder, train_cl, device)
    va_emb,  va_lbl  = extract_embeddings(encoder, val_cl,   device)
    te_emb,  te_lbl  = extract_embeddings(encoder, test_cl,  device)

    # ── Stage 1: cluster ───────────────────────────────────
    imputer = ClusterGuidedImputation(seed=seed)
    imputer.fit(tr_emb)

    # ── Stage 2: impute ────────────────────────────────────
    def _impute_all(ed):
        return {i: imputer.impute(ed[i]) for i in tqdm(ed, desc="Imputing",
                                                        leave=False)}
    print("\nStage 2 — Imputing train …")
    c_tr = _impute_all(tr_emb)
    print("Stage 2 — Imputing val …")
    c_va = _impute_all(va_emb)
    print("Stage 2 — Imputing test …")
    c_te = _impute_all(te_emb)

    # ── Stage 3: train task head ───────────────────────────
    model = UL4M4Model(encoder).to(device)
    model = train_task(model, c_tr, tr_lbl, c_va, va_lbl, device)

    # ── Evaluate ───────────────────────────────────────────
    metrics = evaluate(model, c_te, te_lbl, device)
    print(f"\nTest metrics: {metrics}")
    return metrics


def main():
    all_metrics = []
    for run in range(NUM_RUNS):
        seed = BASE_SEED + run * 100
        print(f"\n{'#'*70}")
        print(f"# RUN {run+1}/{NUM_RUNS}  (seed={seed})")
        print(f"{'#'*70}")
        m = run_once(seed)
        all_metrics.append(m)

    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"channels={AE_CHANNELS}, ps: {PS}, pm: {PM}, lr:{TASK_LR} ")
    print(f"{'#'*70}")
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/ul4m4_ae-pm_{PM}-ps_{PS}-k_{K_CLUSTERS}-H_{FUSE_HEADS}-L_{FUSE_LAYERS}-TLr_{TASK_LR}_.json", "w") as f:
        json.dump({"config": {
            "PM": PM, "PS": PS, "K_CLUSTERS": K_CLUSTERS,
            "AE_LATENT_DIM": AE_LATENT_DIM, "AE_EPOCHS": AE_EPOCHS,
            "TASK_EPOCHS": TASK_EPOCHS, "FUSE_OUT_DIM": FUSE_OUT_DIM,
        }, "runs": all_metrics}, f, indent=2)


if __name__ == "__main__":
    main()