"""
edf_dataset.py  –  Sleep-EDF dataset for FArms / Flower

Mirrors PEPSY's EDF benchmark so results are directly comparable to
Table 1 (EDF rows) of the PEPSY paper.

Missing simulation — PEPSY semantics (from core.py local_missing_setup):
    pm  – fraction of modalities that are "missing leads" for a dataset
    ps  – fraction of samples that miss each of those leads

    PEPSY's local_missing_setup (core.py) does:
      1. Pick floor(K * pm) modalities as "missing leads", seeded by client seed.
      2. For each missing lead i (except the LAST), shuffle sample indices
         and mark the first ps fraction as missing that lead.
      3. For the LAST missing lead, exclude any sample that is already missing
         from ALL previous leads simultaneously (i.e. subtract the intersection).
         This prevents samples from having ALL modalities missing.

═══════════════════════════════════════════════════════════════════════════════
BUG-FIX LOG  (relative to the version you uploaded)
═══════════════════════════════════════════════════════════════════════════════

BUG 5  ──  Missing simulation doesn't match PEPSY's last-lead constraint
  PEPSY's core.py local_missing_setup prevents any sample from having ALL
  missing leads simultaneously by subtracting the intersection of already-
  missing samples from the candidate pool for the last missing lead.
  The previous EDFClientDataset._compute_missing_masks independently shuffled
  for each lead, allowing some samples to receive ALL missing modalities.
  With the sequential simulator, such a sample has c0[b]=zeros (the "all
  missing → stays zero" branch) and then all subsequent ck[b] are simulated
  from zeros, producing meaningless embeddings that hurt training.
  Fix: replicate PEPSY's last-lead exclusion logic exactly.

BUG 6  ──  make_comprehensive_loader: pm=1.0 override comment was misleading
  The old code set pm=1.0 "to make all modalities missing leads" but then
  immediately overrode the masks.  This was functionally correct but wasteful
  and confusing.  Simplified to build masks directly without the pm=1.0 hack.

All other components (make_val_loader, make_test_loader, partition helpers,
_EVAL_SEED, comprehensive loader strategy) were correct and are unchanged.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ── paths ──────────────────────────────────────────────────────────────────────
EDF_DATA_ROOT = "/Documents/py-ipynb/3-EDF/data/SLEEP_EDF/final_data"

N_MODALITIES  = 5
N_CLASSES     = 5
CROP_LENGTH   = 250
CLASS_WEIGHTS = [1.0, 1.0, 0.5, 1.5, 1.0]

LABEL2IDX = {'W': 0, '1': 1, '2': 2, '3': 3, '4': 3, 'R': 4}

# Fixed base seed for all server-side evaluation (val + test).
# Using the same seed for both loaders guarantees val_acc and test_acc
# are evaluated under identical missing patterns and are directly comparable.
_EVAL_SEED = 0


# ── base dataset ───────────────────────────────────────────────────────────────

class EDFBaseDataset(Dataset):
    def __init__(self, root: str, train: bool = True, crop_length: int = CROP_LENGTH):
        self.root        = root
        self.train       = train
        self.crop_length = crop_length

        with open(os.path.join(root, "metadata.json")) as f:
            self.labels = json.load(f)

        split_file = "train_metadata.json" if train else "test_metadata.json"
        with open(os.path.join(root, split_file)) as f:
            self.indices = json.load(f)

        self.y = np.array([LABEL2IDX[self.labels[str(oid)]] for oid in self.indices])

    def _interpolate(self, x: np.ndarray, target_len: int) -> np.ndarray:
        old_len = x.shape[0]
        return np.interp(np.linspace(0, old_len, target_len), np.arange(old_len), x)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=1, keepdims=True)
        std  = np.std(x,  axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        x = np.load(os.path.join(self.root, f"{orig_idx}.npy"))
        x = x[:-2, :].transpose(1, 0)   # (L, 5)

        x_new = np.zeros((self.crop_length, x.shape[1]), dtype=np.float32)
        for c in range(x.shape[1]):
            x_new[:, c] = self._interpolate(x[:, c], self.crop_length)
        x = x_new

        if x.shape[0] >= self.crop_length:
            x = x[:self.crop_length, :].T
        else:
            x = np.pad(x, ((0, self.crop_length - x.shape[0]), (0, 0)),
                       mode='constant', constant_values=-1).T

        x = self._standardize(x.astype(np.float32))
        y = LABEL2IDX[self.labels[str(orig_idx)]]
        return x.astype(np.float32), y


# ── missing-aware client dataset ───────────────────────────────────────────────

class EDFClientDataset(Dataset):
    """
    Wraps EDFBaseDataset for a single federated client.

    Missing simulation replicates PEPSY's local_missing_setup (core.py) exactly,
    including the last-lead intersection exclusion (BUG 5 FIX).

    PEPSY algorithm:
      Step 1: randomly select floor(K * pm) modalities as "missing leads".
      Step 2: for each lead except the last:
                shuffle sample indices (seeded), mark first ps% as missing.
      Step 3: for the LAST lead:
                remove from the candidate pool any sample that is already
                in the intersection of ALL previous missing-sample sets
                (prevents any sample from having ALL modalities missing).
                Then shuffle remaining candidates and mark first ps% as missing.
    """

    def __init__(self, base_dataset: EDFBaseDataset,
                 sample_indices: list,
                 pm: float = 0.0,
                 ps: float = 0.0,
                 seed: int = 0):
        self.base    = base_dataset
        self.indices = sample_indices
        self.pm      = pm
        self.ps      = ps
        self.seed    = seed
        self.n_mod   = N_MODALITIES
        self.missing_masks = self._compute_missing_masks()

    def _compute_missing_masks(self) -> dict:
        """
        Returns {local_idx: [list of missing modality indices]}.

        BUG 5 FIX: replicates PEPSY core.py local_missing_setup, including
        the last-lead exclusion that prevents all-missing samples.
        """
        n_missing_mods = max(1, int(self.n_mod * self.pm))

        # Step 1: which modalities are missing leads
        rng_mod      = random.Random(self.seed)
        all_mods     = list(range(self.n_mod))
        missing_mods = sorted(rng_mod.sample(all_mods, n_missing_mods))

        local_indices = list(range(len(self.indices)))
        missing_masks: dict = {}

        # Per-lead missing sample sets (for the intersection calculation)
        missing_sets_per_lead: list = []  # list of sets of local_idx

        for lead_rank, mod in enumerate(missing_mods):
            rng_samp  = random.Random(self.seed + lead_rank)
            n_missing = int(len(local_indices) * self.ps)

            is_last_lead = (lead_rank == len(missing_mods) - 1)

            if is_last_lead and len(missing_sets_per_lead) > 0:
                # BUG 5 FIX: exclude samples that are already missing from
                # ALL previous leads simultaneously (PEPSY last-lead logic).
                # intersection = samples missing from every prior lead
                intersection = missing_sets_per_lead[0].copy()
                for s in missing_sets_per_lead[1:]:
                    intersection = intersection.intersection(s)
                valid_indices = [i for i in local_indices if i not in intersection]
                shuffled = sorted(valid_indices, key=lambda _: rng_samp.random())
            else:
                shuffled = sorted(local_indices, key=lambda _: rng_samp.random())

            missing_for_this_lead = set(shuffled[:n_missing])
            missing_sets_per_lead.append(missing_for_this_lead)

            for local_idx in missing_for_this_lead:
                missing_masks.setdefault(local_idx, []).append(mod)

        return missing_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        orig_idx = self.indices[local_idx]
        x, y     = self.base[orig_idx]

        missing_mods = self.missing_masks.get(local_idx, [])
        x = x.copy()
        for mod in missing_mods:
            x[mod, :] = -1.0

        has_modality = torch.tensor([x[m, 0] != -1 for m in range(self.n_mod)],
                                    dtype=torch.bool)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), has_modality


# ── federated data loading ─────────────────────────────────────────────────────

def make_federated_loaders(
    root:          str   = EDF_DATA_ROOT,
    num_clients:   int   = 32,
    partition_id:  int   = 0,
    pm:            float = 0.2,
    ps:            float = 0.2,
    iid:           bool  = True,
    seed:          int   = 0,
    batch_size:    int   = 128,
    alpha:         float = 0.5,
    val_fraction:  float = 0.1,
) -> tuple:
    train_base = EDFBaseDataset(root, train=True)

    if iid:
        indices = _iid_partition(train_base, num_clients, seed)[partition_id]
    else:
        indices = _dirichlet_partition(train_base, num_clients, N_CLASSES,
                                       alpha, seed)[partition_id]

    rng      = random.Random(seed + partition_id)
    rng.shuffle(indices)
    val_size  = max(1, int(len(indices) * val_fraction))
    val_idx   = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = EDFClientDataset(train_base, train_idx, pm=pm, ps=ps,
                                seed=seed + partition_id)
    val_ds   = EDFClientDataset(train_base, val_idx, pm=pm, ps=ps,
                                seed=seed + partition_id + 10000)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, drop_last=False)
    return train_loader, val_loader


def make_comprehensive_loader(
    base_dataset:   EDFBaseDataset,
    sample_indices: list,
    pm:             float,
    ps:             float,
    batch_size:     int  = 128,
    base_seed:      int  = _EVAL_SEED,
) -> DataLoader:
    """
    Creates a loader where EVERY modality gets a turn as the sole missing lead,
    each covering ps fraction of the samples.

    BUG 6 FIX: builds missing masks directly (no longer uses pm=1.0 trick).
    For K=5 with pm=0.2, this is identical to what a single-seed loader would
    give for any ONE modality — but we do it for ALL K modalities and concatenate,
    ensuring comprehensive coverage of all missing patterns.

    Why this is needed:
      A single-seed loader with pm=0.2 and K=5 always picks exactly ONE modality
      as the missing lead.  Different seeds pick different modalities.  The server
      and clients would then be evaluated on different missing patterns → gap.
      This loader covers all K modalities equally.
    """
    sub_datasets = []
    for mod in range(N_MODALITIES):
        ds = EDFClientDataset.__new__(EDFClientDataset)
        ds.base    = base_dataset
        ds.indices = sample_indices
        ds.pm      = pm
        ds.ps      = ps
        ds.seed    = base_seed + mod
        ds.n_mod   = N_MODALITIES

        # Build mask: only 'mod' is the missing lead, ps fraction of samples
        local_indices = list(range(len(sample_indices)))
        rng = random.Random(base_seed + mod)
        shuffled  = sorted(local_indices, key=lambda _: rng.random())
        n_missing = int(len(shuffled) * ps)
        ds.missing_masks = {local_idx: [mod] for local_idx in shuffled[:n_missing]}

        sub_datasets.append(ds)

    combined = ConcatDataset(sub_datasets)
    return DataLoader(combined, batch_size=batch_size,
                      shuffle=False, num_workers=0)


def make_val_loader(
    root:         str   = EDF_DATA_ROOT,
    pm:           float = 0.2,
    ps:           float = 0.2,
    seed:         int   = _EVAL_SEED,
    batch_size:   int   = 128,
    val_fraction: float = 0.1,
) -> DataLoader:
    """
    Global validation loader for server-side best-model selection.

    Uses a comprehensive missing simulation (all K modalities covered equally)
    so the server val accuracy reflects generalisation across ALL missing
    patterns, not just whichever modality a single seed happened to pick.

    seed=_EVAL_SEED must be identical to make_test_loader so val_acc and
    test_acc are evaluated under the same missing-pattern structure and are
    directly comparable for model selection and reporting.
    """
    train_base  = EDFBaseDataset(root, train=True)
    all_train   = list(range(len(train_base)))
    rng         = random.Random(seed)
    rng.shuffle(all_train)
    val_size    = max(1, int(len(all_train) * val_fraction))
    val_indices = all_train[:val_size]

    return make_comprehensive_loader(
        train_base, val_indices, pm=pm, ps=ps,
        batch_size=batch_size, base_seed=seed,
    )


def make_test_loader(
    root:       str   = EDF_DATA_ROOT,
    pm:         float = 0.2,
    ps:         float = 0.2,
    seed:       int   = _EVAL_SEED,
    batch_size: int   = 128,
) -> DataLoader:
    """
    Held-out test loader.  Uses the same comprehensive simulation as
    make_val_loader so val_acc and test_acc are directly comparable.
    """
    test_base = EDFBaseDataset(root, train=False)
    all_idx   = list(range(len(test_base)))

    return make_comprehensive_loader(
        test_base, all_idx, pm=pm, ps=ps,
        batch_size=batch_size, base_seed=seed,
    )


# ── partition helpers ──────────────────────────────────────────────────────────

def _iid_partition(dataset, num_clients, seed):
    rng    = random.Random(seed)
    labels = np.unique(dataset.y)
    clients = [[] for _ in range(num_clients)]
    for label in labels:
        idx = np.where(dataset.y == label)[0].tolist()
        rng.shuffle(idx)
        for i, sp in enumerate(np.array_split(idx, num_clients)):
            clients[i].extend(sp.tolist())
    return clients


def _dirichlet_partition(dataset, num_clients, n_classes, alpha, seed):
    np.random.seed(seed)
    clients = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx    = np.where(dataset.y == c)[0]
        props  = np.random.dirichlet(alpha * np.ones(num_clients))
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        cur = 0
        for k, sz in enumerate(splits):
            clients[k].extend(idx[cur: cur + sz].tolist())
            cur += sz
    return clients


if __name__ == "__main__":
    ds = EDFBaseDataset(EDF_DATA_ROOT, train=True)
    print(f"Train size: {len(ds)}")
    x, y = ds[0]
    print(f"  x shape: {x.shape}, y: {y}")

    # Verify BUG 5 fix: no sample should have ALL modalities missing
    client_ds = EDFClientDataset(ds, list(range(100)), pm=0.4, ps=0.6, seed=0)
    x, y, has_mod = client_ds[0]
    print(f"  client x shape: {x.shape}, y: {y}, has_modality: {has_mod}")
    all_missing = sum(1 for i in range(len(client_ds))
                      if not any(client_ds[i][2]))
    print(f"  Samples with ALL modalities missing: {all_missing}  (should be 0)")
    missing_counts = {i: sum(1 for v in client_ds.missing_masks.values() if i in v)
                      for i in range(N_MODALITIES)}
    print(f"  Missing counts per modality: {missing_counts}")