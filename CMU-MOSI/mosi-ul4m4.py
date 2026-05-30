"""
Cluster-Guided Iterative Imputation for Missing Modalities
Implementation of the method from via-ul.txt

This uses the frozen encoders and reuses the fusion/task head from wArms.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import frozen encoders from the wArms implementation
from transformers import BertModel, BertTokenizer, Wav2Vec2FeatureExtractor, WavLMModel
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image

# Import the MOSI dataset
from mosi_reg import MOSIDatasetRegression

DPATH = "/home"
DpO = 0.2
FuseODim = 32
FuseLyN = 1
FuseH = 1
TrainOnComplete = True

# ============================================================================
# FROZEN ENCODERS (same as wArms)
# ============================================================================

class FrozenTextEncoder(nn.Module):
    """Frozen BERT encoder for text"""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, texts):
        with torch.no_grad():
            encoding = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            )
            encoding = {k: v.to(next(self.model.parameters()).device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
            return outputs.last_hidden_state[:, 0, :]

class FrozenAudioEncoder(nn.Module):
    """Frozen WavLM-Large encoder for audio"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, audio_batch):
        with torch.no_grad():
            processed = []
            for audio in audio_batch:
                if isinstance(audio, torch.Tensor):
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                    waveform = audio.squeeze().cpu().numpy()
                else:
                    waveform = audio
                
                if waveform.ndim > 1:
                    waveform = waveform.flatten()
                
                inputs = self.feature_extractor(
                    waveform, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=False
                )
                processed.append(inputs.input_values.squeeze(0))
            
            max_len = max(p.shape[-1] for p in processed)
            padded = torch.stack([
                F.pad(p, (0, max_len - p.shape[-1])) for p in processed
            ]).to(next(self.model.parameters()).device)
            
            outputs = self.model(padded)
            return outputs.last_hidden_state.mean(dim=1)

class FrozenVideoEncoder(nn.Module):
    """Frozen CLIP Vision encoder for video frames"""
    def __init__(self, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, video_batch):
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            all_features = []
            
            for idx, frames in enumerate(video_batch):
                if len(frames) == 0:
                    frame_features = torch.zeros(self.num_frames, 768)
                else:
                    if len(frames) >= self.num_frames:
                        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                    else:
                        repeat_factor = (self.num_frames + len(frames) - 1) // len(frames)
                        indices = np.tile(np.arange(len(frames)), repeat_factor)[:self.num_frames]
                    
                    frame_features_list = []
                    for i in indices:
                        frame = frames[i][:, :, ::-1]  # BGR → RGB
                        pil_image = Image.fromarray(frame.astype('uint8'))
                        
                        inputs = self.processor(images=pil_image, return_tensors="pt")
                        pixel_values = inputs["pixel_values"].to(device)
                        
                        outputs = self.model(pixel_values=pixel_values)
                        frame_feat = outputs.pooler_output
                        frame_features_list.append(frame_feat.squeeze(0))
                    
                    frame_features = torch.stack(frame_features_list)
                
                video_feature = frame_features.mean(dim=0)
                all_features.append(video_feature)
            
            video_feats = torch.stack(all_features).to(device)
            
            for idx, frames in enumerate(video_batch):
                if len(frames) == 0:
                    video_feats[idx] = 0.0
            
            return video_feats


# ============================================================================
# FUSION MODULE (reused from wArms)
# ============================================================================
#_wDp
class FusionModule(nn.Module):
    """Multi-Head Self-Attention fusion with dropout"""
    def __init__(self, input_dims, output_dim=FuseODim, num_heads=FuseH, num_layers=FuseLyN, dropout_p=DpO):
        super().__init__()
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads,
                                  dropout=dropout_p, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        # Dropout applied to attention output before residual add+norm,
        # following the standard transformer block convention.
        self.dropout = nn.Dropout(p=dropout_p)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, *features):
        projected = [self.modality_projections[i](feat) for i, feat in enumerate(features)]
        modality_sequence = torch.stack(projected, dim=1)
        
        attn_output = modality_sequence
        for attn_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn_layer(attn_output, attn_output, attn_output)
            attn_output = layer_norm(attn_output + self.dropout(attn_out))
        
        fused = attn_output.mean(dim=1)
        return self.relu(self.output_proj(fused))

#_withSpeededUp_KM&++
class ClusterGuidedImputation:
    """
    Implements the cluster-guided iterative imputation method from via-ul.txt
    """
    def __init__(self, text_dim=768, audio_dim=1024, video_dim=768, k_clusters=10, 
                 max_kmeans_iters=100, device='cuda', seed=42):
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.k_clusters = k_clusters
        self.max_kmeans_iters = max_kmeans_iters
        self.device = device
        self.seed = seed
        
        # Modality information
        self.modality_names = ['text', 'audio', 'video']
        self.modality_dims = [text_dim, audio_dim, video_dim]
        
        # Will be computed during fit
        self.cluster_centres = None  # List of k centres, each is dict {modality: embedding}
        self.normalization_stats = None  # Dict {modality: (mean, std)}
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _compute_normalization_stats(self, embeddings_dict):
        """
        Compute modality-specific normalization statistics.
        
        Args:
            embeddings_dict: Dict mapping sample_idx -> dict of {modality: embedding}
        
        Returns:
            Dict {modality: (mean, std)}
        """
        stats = {}
        
        for mod_idx, mod_name in enumerate(self.modality_names):
            # Collect all available embeddings for this modality
            available_embeddings = []
            for sample_idx, sample_dict in embeddings_dict.items():
                if mod_name in sample_dict and sample_dict[mod_name] is not None:
                    available_embeddings.append(sample_dict[mod_name])
            
            if len(available_embeddings) > 0:
                available_embeddings = torch.stack(available_embeddings)
                mean = available_embeddings.mean(dim=0)
                std = available_embeddings.std(dim=0)
                # Avoid division by zero
                std = torch.where(std > 1e-8, std, torch.ones_like(std))
                stats[mod_name] = (mean, std)
            else:
                # Shouldn't happen if dataset contains all modalities
                dim = self.modality_dims[mod_idx]
                stats[mod_name] = (
                    torch.zeros(dim).to(self.device),
                    torch.ones(dim).to(self.device)
                )
        
        return stats
    
    def _normalize_embedding(self, embedding, modality):
        """Apply z-score normalization to an embedding"""
        mean, std = self.normalization_stats[modality]
        return (embedding - mean) / std
    
    def _partial_modality_distance(self, sample1_dict, sample2_dict, use_normalized=True):
        """
        Compute partial-modality distance between two samples.
        
        Args:
            sample1_dict: Dict {modality: embedding or normalized_embedding}
            sample2_dict: Dict {modality: embedding or normalized_embedding}
            use_normalized: Whether inputs are already normalized
        
        Returns:
            Distance (float), or inf if no shared modalities
        """
        shared_modalities = []
        for mod_name in self.modality_names:
            if (mod_name in sample1_dict and sample1_dict[mod_name] is not None and
                mod_name in sample2_dict and sample2_dict[mod_name] is not None):
                shared_modalities.append(mod_name)
        
        if len(shared_modalities) == 0:
            return float('inf')
        
        total_squared_error = 0.0
        for mod_name in shared_modalities:
            emb1 = sample1_dict[mod_name]
            emb2 = sample2_dict[mod_name]
            
            if not use_normalized:
                emb1 = self._normalize_embedding(emb1, mod_name)
                emb2 = self._normalize_embedding(emb2, mod_name)
            
            squared_diff = ((emb1 - emb2) ** 2).sum().item()
            
            # Get dimensionality for this modality
            mod_idx = self.modality_names.index(mod_name)
            d_m = self.modality_dims[mod_idx]
            
            # Divide by dimensionality
            total_squared_error += squared_diff / d_m
        
        # Divide by number of shared modalities
        distance = np.sqrt(total_squared_error / len(shared_modalities))
        return distance
    
    def _initialize_centres(self, embeddings_dict):
        """
        Initialize k cluster centres using k-means++ seeding.

        Algorithm:
          1. Pick the first centre uniformly at random from all training samples.
          2. For each subsequent centre, compute for every sample the distance
             d(x, nearest chosen centre) using the partial-modality distance on
             normalised embeddings.  Sample the next centre with probability
             proportional to d².  This spreads centres far apart, reducing the
             chance of empty clusters and speeding convergence.
          3. Repeat until K centres are chosen.

        Each chosen centre inherits exactly the modalities present in its
        initialising sample, consistent with the paper's specification.
        """
        sample_indices = list(embeddings_dict.keys())
        N = len(sample_indices)

        # Normalise once for distance computations during seeding
        norm_samples = {
            sid: {
                m: self._normalize_embedding(emb, m)
                for m, emb in embeddings_dict[sid].items()
                if emb is not None
            }
            for sid in sample_indices
        }

        # Step 1 — first centre chosen uniformly at random
        first_idx = sample_indices[np.random.randint(N)]
        chosen_indices = [first_idx]

        # d2[i] = squared distance from sample i to its nearest chosen centre so far
        d2 = np.full(N, np.inf)

        def _update_d2(new_centre_sid):
            nc_norm = norm_samples[new_centre_sid]
            for n_idx, sid in enumerate(sample_indices):
                d = self._partial_modality_distance(
                    norm_samples[sid], nc_norm, use_normalized=True
                )
                # treat inf as large finite so the probability stays well-defined
                if d == float('inf'):
                    d = 1e9
                d2[n_idx] = min(d2[n_idx], d * d)

        _update_d2(first_idx)

        # Steps 2…K — pick remaining centres proportional to d²
        for _ in range(self.k_clusters - 1):
            total = d2.sum()
            probs = d2 / total if total > 0 else np.ones(N) / N
            next_n_idx = np.random.choice(N, p=probs)
            next_sid = sample_indices[next_n_idx]
            chosen_indices.append(next_sid)
            _update_d2(next_sid)

        # Build centre dicts from chosen samples (unnormalised embeddings)
        centres = []
        for idx in chosen_indices:
            centre = {
                mod_name: emb.clone()
                for mod_name, emb in embeddings_dict[idx].items()
                if emb is not None
            }
            centres.append(centre)

        return centres
    
    def _kmeans_update_step(self, embeddings_dict, assignments):
        """
        Update cluster centres by averaging embeddings of assigned samples.
        For each centre and modality, compute mean of UNNORMALIZED features.
        """
        new_centres = []
        
        for cluster_idx in range(self.k_clusters):
            # Find samples assigned to this cluster
            assigned_samples = [idx for idx, c_idx in assignments.items() if c_idx == cluster_idx]
            
            centre = {}
            for mod_name in self.modality_names:
                # Collect embeddings for this modality from assigned samples
                embeddings_for_mod = []
                for sample_idx in assigned_samples:
                    if (mod_name in embeddings_dict[sample_idx] and 
                        embeddings_dict[sample_idx][mod_name] is not None):
                        embeddings_for_mod.append(embeddings_dict[sample_idx][mod_name])
                
                if len(embeddings_for_mod) > 0:
                    # Average the unnormalized embeddings
                    centre[mod_name] = torch.stack(embeddings_for_mod).mean(dim=0)
                # If no samples have this modality, centre[mod_name] is not defined (missing)
            
            new_centres.append(centre)
        
        return new_centres
    
    def _centre_centre_distances(self, normalized_centres):
        """
        Compute the full K×K matrix of inter-centre distances (partial-modality distance).
        Used by the Elkan variant to prune unnecessary sample-centre distance evaluations.
        Returns a numpy array of shape (K, K).
        """
        K = len(normalized_centres)
        D = np.full((K, K), 0.0)
        for a in range(K):
            for b in range(a + 1, K):
                d = self._partial_modality_distance(
                    normalized_centres[a], normalized_centres[b], use_normalized=True
                )
                D[a, b] = d
                D[b, a] = d
        return D

    def fit(self, embeddings_dict):
        """
        Perform k-means clustering using the Elkan variant.

        The Elkan algorithm maintains:
          - upper[i]   : upper bound on d(sample_i, its assigned centre)
          - lower[i,j] : lower bound on d(sample_i, centre_j)
        It uses the triangle inequality to skip centre-distance evaluations
        that cannot improve the current assignment, making it significantly
        faster than Lloyd's algorithm while producing identical results.

        Args:
            embeddings_dict: Dict mapping sample_idx -> dict of {modality: embedding}
        """
        print(f"\n{'='*80}")
        print("STAGE 1: CLUSTERING WITH PARTIAL-MODALITY DISTANCE (Elkan variant)")
        print(f"{'='*80}")
        print(f"Number of clusters: {self.k_clusters}")
        print(f"Number of samples:  {len(embeddings_dict)}")

        # ── Normalisation ────────────────────────────────────────────────────
        print("Computing normalization statistics...")
        self.normalization_stats = self._compute_normalization_stats(embeddings_dict)

        # Pre-normalise all sample embeddings once
        sample_ids = list(embeddings_dict.keys())
        N = len(sample_ids)
        K = self.k_clusters

        normalized_samples = {}
        for sid in sample_ids:
            normalized_samples[sid] = {
                m: self._normalize_embedding(emb, m)
                for m, emb in embeddings_dict[sid].items()
                if emb is not None
            }

        # ── Initialisation ───────────────────────────────────────────────────
        print("Initializing cluster centres...")
        centres = self._initialize_centres(embeddings_dict)

        # Normalise initial centres
        def _norm_centres(raw_centres):
            nc = []
            for c in raw_centres:
                nc.append({m: self._normalize_embedding(emb, m)
                            for m, emb in c.items() if emb is not None})
            return nc

        norm_centres = _norm_centres(centres)

        # Compute initial exact distances  d(x_i, c_j)  for all i, j
        # lower[i, j] = lower bound on d(sample_i, centre_j)
        # upper[i]    = upper bound on d(sample_i, assigned centre)
        lower = np.full((N, K), 0.0)
        upper = np.full(N, np.inf)
        assignments = np.zeros(N, dtype=int)

        print("Computing initial assignments...")
        for n_idx, sid in enumerate(sample_ids):
            best_d = np.inf
            best_c = 0
            for c_idx in range(K):
                d = self._partial_modality_distance(
                    normalized_samples[sid], norm_centres[c_idx], use_normalized=True
                )
                lower[n_idx, c_idx] = d
                if d < best_d:
                    best_d = d
                    best_c = c_idx
            assignments[n_idx] = best_c
            upper[n_idx] = best_d

        # Flag: upper bound may be stale (needs re-computation before use)
        r = np.ones(N, dtype=bool)   # True = upper bound is tight / needs refresh

        # ── Elkan iterations ─────────────────────────────────────────────────
        print("Running Elkan k-means...")
        for iter_idx in range(self.max_kmeans_iters):

            # Step 1 – centre-centre distances & half-distances
            cc_dist = self._centre_centre_distances(norm_centres)
            s = 0.5 * np.min(
                cc_dist + np.eye(K) * np.inf, axis=1
            )   # s[c] = 0.5 * d(c, nearest other centre)

            # Step 2 – skip samples whose upper bound ≤ s[assigned centre]
            changed_any = False
            for n_idx, sid in enumerate(sample_ids):
                c_cur = assignments[n_idx]
                if upper[n_idx] <= s[c_cur]:
                    continue   # triangle inequality guarantees assignment won't change

                for c_idx in range(K):
                    if c_idx == c_cur:
                        continue
                    # Elkan condition: skip if upper ≤ lower or upper ≤ half cc_dist
                    if upper[n_idx] <= lower[n_idx, c_idx]:
                        continue
                    if upper[n_idx] <= 0.5 * cc_dist[c_cur, c_idx]:
                        continue

                    # Tighten upper bound if stale
                    if r[n_idx]:
                        d_cur = self._partial_modality_distance(
                            normalized_samples[sid],
                            norm_centres[c_cur],
                            use_normalized=True
                        )
                        upper[n_idx] = d_cur
                        lower[n_idx, c_cur] = d_cur
                        r[n_idx] = False
                    else:
                        d_cur = upper[n_idx]

                    if d_cur <= lower[n_idx, c_idx]:
                        continue

                    # Must compute exact distance to c_idx
                    d_new = self._partial_modality_distance(
                        normalized_samples[sid],
                        norm_centres[c_idx],
                        use_normalized=True
                    )
                    lower[n_idx, c_idx] = d_new

                    if d_new < d_cur:
                        assignments[n_idx] = c_idx
                        upper[n_idx] = d_new
                        c_cur = c_idx
                        changed_any = True

            # Step 3 – recompute centres (unnormalised means) and measure drift
            assignments_dict = {sid: int(assignments[n_idx])
                                for n_idx, sid in enumerate(sample_ids)}
            new_centres = self._kmeans_update_step(embeddings_dict, assignments_dict)
            new_norm_centres = _norm_centres(new_centres)

            # Centre drift distances  δ[c]  (used to update bounds)
            delta = np.array([
                self._partial_modality_distance(
                    norm_centres[c], new_norm_centres[c], use_normalized=True
                )
                for c in range(K)
            ])

            # Step 4 – update bounds
            for n_idx in range(N):
                c_cur = assignments[n_idx]
                # Lower bounds shrink by at most δ[c]
                lower[n_idx] = np.maximum(0.0, lower[n_idx] - delta)
                # Upper bound grows by δ[assigned centre]
                upper[n_idx] += delta[c_cur]
                r[n_idx] = True   # upper is now potentially stale

            norm_centres = new_norm_centres
            centres = new_centres

            converged = not changed_any or np.all(delta < 1e-6)
            if (iter_idx + 1) % 10 == 0:
                print(f"  Iteration {iter_idx + 1}/{self.max_kmeans_iters}"
                      f"  |  max centre drift: {delta.max():.2e}"
                      f"  |  assignments changed: {changed_any}")
            if converged:
                print(f"  Converged at iteration {iter_idx + 1}")
                break

        self.cluster_centres = centres

        # Final assignment dict (str keys → cluster index)
        final_assignments = {sid: int(assignments[n_idx])
                             for n_idx, sid in enumerate(sample_ids)}

        print(f"\nCluster statistics:")
        cluster_counts = np.bincount(assignments, minlength=K)
        for c_idx, centre in enumerate(centres):
            available_mods = [m for m in self.modality_names if m in centre]
            print(f"  Cluster {c_idx:>2d}: {', '.join(available_mods):<25s}"
                  f"  ({cluster_counts[c_idx]} samples)")

        print(f"{'='*80}\n")
        return final_assignments
    
    def impute_sample(self, sample_embeddings):
        """
        Impute missing modalities for a single sample using iterative greedy completion.
        
        Args:
            sample_embeddings: Dict {modality: embedding or None}
        
        Returns:
            Complete dict {modality: embedding} with all modalities filled
        """
        # Identify missing modalities
        missing_modalities = [m for m in self.modality_names 
                             if m not in sample_embeddings or sample_embeddings[m] is None]
        
        if len(missing_modalities) == 0:
            return sample_embeddings  # Already complete
        
        # Create initial candidates from all cluster centres
        candidates = []
        for centre_idx, centre in enumerate(self.cluster_centres):
            candidate = copy.deepcopy(sample_embeddings)
            
            # Fill missing modalities from this centre
            for mod_name in missing_modalities:
                if mod_name in centre:
                    candidate[mod_name] = centre[mod_name].clone()
            
            candidates.append(candidate)
        
        # Track which candidates are still available (by index)
        available_candidate_indices = set(range(len(candidates)))
        
        # Iteratively select best candidate and fill missing modalities
        result = copy.deepcopy(sample_embeddings)
        
        while len(missing_modalities) > 0 and len(available_candidate_indices) > 0:
            # Compute distances between all available candidates and all centres
            min_distance = float('inf')
            best_candidate_idx = None
            
            for cand_idx in available_candidate_indices:
                candidate = candidates[cand_idx]
                for centre in self.cluster_centres:
                    distance = self._partial_modality_distance(
                        candidate, centre, use_normalized=False
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_candidate_idx = cand_idx
            
            if best_candidate_idx is None:
                break
            
            # Get the best candidate and remove its index from available set
            best_candidate = candidates[best_candidate_idx]
            available_candidate_indices.remove(best_candidate_idx)
            
            # Fill missing modalities from selected candidate
            for mod_name in missing_modalities[:]:  # Copy list to modify during iteration
                if mod_name in best_candidate and best_candidate[mod_name] is not None:
                    result[mod_name] = best_candidate[mod_name].clone()
                    missing_modalities.remove(mod_name)
            
            # If no more missing modalities, we're done
            if len(missing_modalities) == 0:
                break
        
        return result

# ============================================================================
# MAIN MODEL WITH CLUSTER IMPUTATION
# ============================================================================

class ClusterImputationModel(nn.Module):
    """
    Model using cluster-guided imputation + fusion + task head
    """
    def __init__(self, text_dim=768, audio_dim=1024, video_dim=768, output_dim=1):
        super().__init__()
        
        # Frozen encoders
        self.text_encoder = FrozenTextEncoder()
        self.audio_encoder = FrozenAudioEncoder()
        self.video_encoder = FrozenVideoEncoder()
        
        # Dimensions
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        
        # Final fusion (trainable)
        self.final_fusion = FusionModule([text_dim, audio_dim, video_dim])
        
        # Task head (trainable)
        # self.regressor = nn.Sequential(
        #     nn.Linear(FuseODim, 128),
        #     nn.ReLU(),
        #     #nn.Dropout(0.2),
        #     nn.Linear(128, output_dim),
        #     nn.Hardtanh(min_val=-3.0, max_val=3.0)
        # )
        self.regressor = nn.Sequential(
            nn.Linear(FuseODim, 1),
            nn.Hardtanh(min_val=-3.0, max_val=3.0)
        )
        
        # Cluster imputation (will be set during training)
        self.cluster_imputer = None
        
        # Count parameters
        fusion_params = sum(p.numel() for p in self.final_fusion.parameters())
        regressor_params = sum(p.numel() for p in self.regressor.parameters())
        frozen_params = sum(p.numel() for p in self.text_encoder.parameters())
        frozen_params += sum(p.numel() for p in self.audio_encoder.parameters())
        frozen_params += sum(p.numel() for p in self.video_encoder.parameters())
        
        print(f"\n{'='*80}")
        print("CLUSTER IMPUTATION MODEL")
        print(f"{'='*80}")
        print(f"  Frozen encoder parameters:        {frozen_params:,}")
        print(f"  Fusion parameters:                {fusion_params:,}")
        print(f"  Task head parameters:             {regressor_params:,}")
        print(f"  Total trainable parameters:       {fusion_params + regressor_params:,}")
        print(f"  Total parameters:                 {frozen_params + fusion_params + regressor_params:,}")
        print(f"{'='*80}\n")
    
    def extract_embeddings_batch(self, text, audio, video, has_text, has_audio, has_video):
        """Extract embeddings for a batch, respecting availability flags"""
        batch_size = len(text)
        device = next(self.parameters()).device
        
        # Initialize with None
        text_feats = [None] * batch_size
        audio_feats = [None] * batch_size
        video_feats = [None] * batch_size
        
        # Extract text embeddings
        text_available_indices = [i for i in range(batch_size) if has_text[i]]
        if text_available_indices:
            text_batch = [text[i] for i in text_available_indices]
            text_embeddings = self.text_encoder(text_batch)
            for idx, embedding in zip(text_available_indices, text_embeddings):
                text_feats[idx] = embedding
        
        # Extract audio embeddings
        audio_available_indices = [i for i in range(batch_size) if has_audio[i]]
        if audio_available_indices:
            audio_batch = [audio[i] for i in audio_available_indices]
            audio_embeddings = self.audio_encoder(audio_batch)
            for idx, embedding in zip(audio_available_indices, audio_embeddings):
                audio_feats[idx] = embedding
        
        # Extract video embeddings
        video_available_indices = [i for i in range(batch_size) if has_video[i]]
        if video_available_indices:
            video_batch = [video[i] for i in video_available_indices]
            video_embeddings = self.video_encoder(video_batch)
            for idx, embedding in zip(video_available_indices, video_embeddings):
                video_feats[idx] = embedding
        
        return text_feats, audio_feats, video_feats
    
    def forward(self, text_feats, audio_feats, video_feats):
        """
        Forward pass assuming all features are complete (after imputation).
        
        Args:
            text_feats: Tensor [batch_size, text_dim]
            audio_feats: Tensor [batch_size, audio_dim]
            video_feats: Tensor [batch_size, video_dim]
        """
        # Fuse all modalities
        fused = self.final_fusion(text_feats, audio_feats, video_feats)
        
        # Predict
        predictions = self.regressor(fused).squeeze(-1)
        
        return predictions


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def collate_fn(batch):
    """Custom collate function"""
    return {
        'name': [item['name'] for item in batch],
        'text': [item['text'] for item in batch],
        'audio': [item['audio'] for item in batch],
        'video': [item['video'] for item in batch],
        'label': torch.stack([item['label'] for item in batch]),
        'has_text': [item['has_text'] for item in batch],
        'has_audio': [item['has_audio'] for item in batch],
        'has_video': [item['has_video'] for item in batch]
    }


def extract_all_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples in the dataset.
    
    Returns:
        embeddings_dict: Dict mapping sample_idx -> {modality: embedding}
        sample_indices_map: Dict mapping sample_idx -> sample_name
    """
    model.eval()
    
    embeddings_dict = {}
    sample_indices_map = {}
    sample_idx = 0
    
    print("Extracting embeddings from dataset...")
    for batch in tqdm(dataloader, desc="Extracting"):
        text = batch['text']
        audio = batch['audio']
        video = batch['video']
        has_text = batch['has_text']
        has_audio = batch['has_audio']
        has_video = batch['has_video']
        names = batch['name']
        
        text_feats, audio_feats, video_feats = model.extract_embeddings_batch(
            text, audio, video, has_text, has_audio, has_video
        )
        
        batch_size = len(text)
        for i in range(batch_size):
            sample_dict = {}
            if text_feats[i] is not None:
                sample_dict['text'] = text_feats[i].detach().cpu()
            if audio_feats[i] is not None:
                sample_dict['audio'] = audio_feats[i].detach().cpu()
            if video_feats[i] is not None:
                sample_dict['video'] = video_feats[i].detach().cpu()
            
            embeddings_dict[sample_idx] = sample_dict
            sample_indices_map[sample_idx] = names[i]
            sample_idx += 1
    
    return embeddings_dict, sample_indices_map


def impute_all_embeddings(embeddings_dict, cluster_imputer, device):
    """
    Impute missing modalities for all samples.
    
    Returns:
        complete_embeddings: Dict mapping sample_idx -> {modality: embedding} (all complete)
    """
    print("\n" + "="*80)
    print("STAGE 2: ITERATIVE GREEDY IMPUTATION")
    print("="*80)
    
    complete_embeddings = {}
    
    for sample_idx in tqdm(embeddings_dict.keys(), desc="Imputing"):
        sample_emb = embeddings_dict[sample_idx]
        complete_emb = cluster_imputer.impute_sample(sample_emb)
        
        # Ensure all modalities are present
        assert 'text' in complete_emb and complete_emb['text'] is not None
        assert 'audio' in complete_emb and complete_emb['audio'] is not None
        assert 'video' in complete_emb and complete_emb['video'] is not None
        
        complete_embeddings[sample_idx] = complete_emb
    
    print(f"{'='*80}\n")
    
    return complete_embeddings


def train_task(model, complete_train_embeddings, train_labels, 
               complete_val_embeddings, val_labels,
               num_epochs=50, lr=1e-3, batch_size=32, device='cuda'):
    """
    Train the task head (fusion + regressor) using complete embeddings.
    """
    print("\n" + "="*80)
    print("TASK TRAINING (Fusion + Head)")
    print("="*80)
    
    model.train()
    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.video_encoder.eval()
    
    # Create tensors from complete embeddings
    train_size = len(complete_train_embeddings)
    train_text = torch.stack([complete_train_embeddings[i]['text'] for i in range(train_size)]).to(device)
    train_audio = torch.stack([complete_train_embeddings[i]['audio'] for i in range(train_size)]).to(device)
    train_video = torch.stack([complete_train_embeddings[i]['video'] for i in range(train_size)]).to(device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).to(device)
    
    val_size = len(complete_val_embeddings)
    val_text = torch.stack([complete_val_embeddings[i]['text'] for i in range(val_size)]).to(device)
    val_audio = torch.stack([complete_val_embeddings[i]['audio'] for i in range(val_size)]).to(device)
    val_video = torch.stack([complete_val_embeddings[i]['video'] for i in range(val_size)]).to(device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).to(device)
    
    # Optimizer for fusion + regressor
    optimizer = torch.optim.Adam(
        list(model.final_fusion.parameters()) + list(model.regressor.parameters()),
        lr=lr
    )
    
    best_val_loss = float('inf')
    best_binary_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        model.text_encoder.eval()
        model.audio_encoder.eval()
        model.video_encoder.eval()
        
        # Mini-batch training
        indices = torch.randperm(train_size)
        epoch_loss = 0.0
        num_batches = 0
        
        for start_idx in range(0, train_size, batch_size):
            end_idx = min(start_idx + batch_size, train_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_text = train_text[batch_indices]
            batch_audio = train_audio[batch_indices]
            batch_video = train_video[batch_indices]
            batch_labels = train_labels_tensor[batch_indices]
            
            optimizer.zero_grad()
            predictions = model(batch_text, batch_audio, batch_video)
            loss = F.mse_loss(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_text, val_audio, val_video)
            val_loss = F.mse_loss(val_predictions, val_labels_tensor).item()

            # binary_preds = [1 if p >= 0 else 0 for p in val_predictions]
            # binary_labels = [1 if l >= 0 else 0 for l in val_labels_tensor]
            # binary_acc = sum(1 for p, l in zip(binary_preds, binary_labels) if p == l) / len(binary_labels) if binary_labels else 0.0
        
        # if binary_acc > best_binary_acc:
        #     print(f"Better model found in epoch {epoch+1}/{num_epochs} | Val Acc: {binary_acc:.3f}")
        #     best_binary_acc = binary_acc
        #     best_model_state = copy.deepcopy(model.state_dict())

        if val_loss < best_val_loss:
            print(f"Better model found in epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.3f}")
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (Val Loss: {best_val_loss:.4f})")
    
    print(f"{'='*80}\n")


def evaluate_model(model, complete_test_embeddings, test_labels, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    
    test_size = len(complete_test_embeddings)
    test_text = torch.stack([complete_test_embeddings[i]['text'] for i in range(test_size)]).to(device)
    test_audio = torch.stack([complete_test_embeddings[i]['audio'] for i in range(test_size)]).to(device)
    test_video = torch.stack([complete_test_embeddings[i]['video'] for i in range(test_size)]).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(test_text, test_audio, test_video)
    
    predictions_np = predictions.cpu().numpy()
    labels_np = test_labels_tensor.cpu().numpy()
    
    mse = mean_squared_error(labels_np, predictions_np)
    mae = mean_absolute_error(labels_np, predictions_np)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(predictions_np, labels_np)
    
    # Binary accuracy
    pred_binary = (predictions_np >= 0).astype(int)
    true_binary = (labels_np >= 0).astype(int)
    binary_acc = (pred_binary == true_binary).mean()
    
    return mse, mae, rmse, corr, binary_acc, predictions_np, labels_np

import os
from sklearn.metrics import f1_score
def calculate_f1_scores(all_preds, all_labels):
    """Calculate F1 micro and macro scores"""
    # Convert continuous predictions to binary (positive/negative sentiment)
    binary_preds = [1 if p >= 0 else 0 for p in all_preds]
    binary_labels = [1 if l >= 0 else 0 for l in all_labels]
    
    f1_micro = f1_score(binary_labels, binary_preds, average='micro')
    f1_macro = f1_score(binary_labels, binary_preds, average='macro')
    
    return f1_micro, f1_macro
    
def visualize_imputation(
    model,
    test_embeddings,          # imputed test embeddings (dict idx→{mod:tensor})
    test_dataset,             # MOSIDatasetRegression for the current config
    audio_dir, video_dir, text_dir, split_file,
    cluster_imputer,
    missing_config,
    run_idx,
    results_dir,
    device,
    batch_size=8,
    tsne_perplexity=30,
    tsne_seed=0,
):
    """
    For each modality that is missing in at least one test sample, produce a t-SNE
    scatter plot comparing:
      • REAL embedding  (from the 100%-complete dataset) — coloured filled circle
      • IMPUTED embedding (from cluster imputation)      — red triangle

    Colours by modality:
      text  → blue  (#2196F3)
      audio → green (#4CAF50)
      video → gold  (#FFC107)

    Saved as:
      text-{missing_config}-{run_idx+1}.png
      audio-{missing_config}-{run_idx+1}.png
      video-{missing_config}-{run_idx+1}.png
    """
    from sklearn.manifold import TSNE

    # ── 1. Identify which samples are missing which modality ─────────────────
    missing_sets = {'text': [], 'audio': [], 'video': []}
    for idx in range(len(test_dataset)):
        has_t, has_a, has_v = test_dataset.modality_availability[idx]
        if not has_t:
            missing_sets['text'].append(idx)
        if not has_a:
            missing_sets['audio'].append(idx)
        if not has_v:
            missing_sets['video'].append(idx)

    # Skip entirely if nothing is missing
    if all(len(v) == 0 for v in missing_sets.values()):
        print("  (no missing modalities in this config – skipping visualisation)")
        return

    # ── 2. Extract REAL embeddings from the 100%-complete dataset ────────────
    complete_test_dataset = MOSIDatasetRegression(
        audio_dir=audio_dir, video_dir=video_dir, text_dir=text_dir,
        split_file=split_file, split='test',
        missing_config='100_text_100_audio_100_video',
        seed=0
    )
    complete_loader = DataLoader(
        complete_test_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    real_embeddings = {}  # idx → {mod: tensor}   (all on CPU)
    real_idx = 0
    for batch in complete_loader:
        text  = batch['text']
        audio = batch['audio']
        video = batch['video']
        has_t = batch['has_text']
        has_a = batch['has_audio']
        has_v = batch['has_video']
        t_feats, a_feats, v_feats = model.extract_embeddings_batch(
            text, audio, video, has_t, has_a, has_v
        )
        for i in range(len(text)):
            real_embeddings[real_idx] = {
                'text':  t_feats[i].detach().cpu() if t_feats[i] is not None else None,
                'audio': a_feats[i].detach().cpu() if a_feats[i] is not None else None,
                'video': v_feats[i].detach().cpu() if v_feats[i] is not None else None,
            }
            real_idx += 1

    # ── 3. Normalization helper (uses training-set stats for consistency) ────
    def normalize(emb, mod_name):
        mean, std = cluster_imputer.normalization_stats[mod_name]
        return ((emb - mean) / std).numpy()

    # ── 4. Per-modality plot ─────────────────────────────────────────────────
    colour_map   = {'text': '#2196F3', 'audio': '#4CAF50', 'video': '#FFC107'}
    modality_label = {'text': 'Text', 'audio': 'Audio', 'video': 'Video'}

    for mod_name, missing_indices in missing_sets.items():
        if len(missing_indices) == 0:
            continue

        # Gather normalised real and imputed vectors for this modality
        real_vecs    = []
        imputed_vecs = []

        for idx in missing_indices:
            r_emb = real_embeddings[idx][mod_name]
            i_emb = test_embeddings[idx][mod_name]   # already a tensor on CPU

            if r_emb is None or i_emb is None:
                continue

            real_vecs.append(normalize(r_emb,    mod_name))
            imputed_vecs.append(normalize(i_emb, mod_name))

        if len(real_vecs) == 0:
            continue

        real_arr    = np.stack(real_vecs)     # (N, D)
        imputed_arr = np.stack(imputed_vecs)  # (N, D)
        combined    = np.concatenate([real_arr, imputed_arr], axis=0)  # (2N, D)

        n_samples = combined.shape[0]
        perp = min(tsne_perplexity, max(5, n_samples // 3 - 1))

        print(f"  Running t-SNE for {mod_name} "
              f"({len(real_vecs)} missing samples, perplexity={perp})...")
        tsne = TSNE(n_components=2, perplexity=perp,
                    random_state=tsne_seed, max_iter=1000)
        proj = tsne.fit_transform(combined)

        proj_real    = proj[:len(real_vecs)]
        proj_imputed = proj[len(real_vecs):]

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(
            proj_real[:, 0], proj_real[:, 1],
            c=colour_map[mod_name], marker='o', s=60, alpha=0.75,
            label=f'Real {modality_label[mod_name]}', edgecolors='none'
        )
        ax.scatter(
            proj_imputed[:, 0], proj_imputed[:, 1],
            c='#F44336', marker='^', s=60, alpha=0.75,
            label=f'Imputed {modality_label[mod_name]}', edgecolors='none'
        )

        ax.set_title(
            f'{modality_label[mod_name]} Embeddings: Real vs Imputed\n'
            f'Config: {missing_config} | Run {run_idx + 1} '
            f'| N={len(real_vecs)}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('t-SNE dim 1', fontsize=11)
        ax.set_ylabel('t-SNE dim 2', fontsize=11)
        ax.legend(fontsize=11, markerscale=1.4)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()

        save_path = os.path.join(
            results_dir,
            f"{mod_name}-{missing_config}-{run_idx + 1}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {save_path}")


def save_results_to_json(results, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(missing_configs=None, num_runs=1, k_clusters=10, num_epochs=50, 
         lr=5e-4, batch_size=32, device='cuda'):
    """
    Main training pipeline for cluster-guided imputation.
    """
    
    # Default configurations
    if missing_configs is None:
        missing_configs = [
            '100_text_100_audio_100_video',  # Complete (baseline)
            '50_text_100_audio_100_video',   # 50% text missing
            'complex_20_20_20_10_10_10_10',  # Complex distribution
        ]
    
    # Dataset paths
    audio_dir = DPATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Audio/WAV_16000/Segmented"
    video_dir = DPATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Video/Segmented"
    text_dir = DPATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Transcript/Segmented"
    split_file = DPATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/mosi_splits-70train.json"
    
    results_dir=f'ul-dp{DpO}-FuseD{FuseODim}-FuseL{FuseLyN}-FuseH{FuseH}-k{k_clusters}-bs{batch_size}-e{num_epochs}-lr{lr}'
    if TrainOnComplete:
        results_dir=f'ul_C-dp{DpO}-FuseD{FuseODim}-FuseL{FuseLyN}-FuseH{FuseH}-k{k_clusters}-bs{batch_size}-e{num_epochs}-lr{lr}'
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    my_seeds = [
        42, 123, 256, 512, 1024,
        2048, 3141, 5000, 7777, 9999,
        12345, 54321, 11111, 22222, 33333,
        44444, 55555, 66666, 77777, 88888,
        99999, 13579, 24680, 31415, 27182,
        16180, 86753, 10101, 20202, 30303
    ]

    recorded_fmicro_scores = []
    for missing_config in missing_configs:
        print(f"\n{'#'*80}")
        print(f"# CONFIGURATION: {missing_config}")
        print(f"{'#'*80}\n")
        
        config_results = []
        
        sum_run = 0.0
        for run_idx in range(num_runs):
            run_seed = my_seeds[run_idx]
            print(f"\n{'='*80}")
            print(f"RUN {run_idx+1}/{num_runs} (seed={run_seed})")
            print(f"{'='*80}\n")
            
            # Create datasets
            train_dataset = MOSIDatasetRegression(
                audio_dir=audio_dir, video_dir=video_dir, text_dir=text_dir,
                split_file=split_file, split='train',
                missing_config=missing_config, seed=run_seed
            )
            if TrainOnComplete:
                train_dataset = MOSIDatasetRegression(
                    audio_dir=audio_dir, video_dir=video_dir, text_dir=text_dir,
                    split_file=split_file, split='train',
                    missing_config="100_text_100_audio_100_video", seed=run_seed
                )
            val_dataset = MOSIDatasetRegression(
                audio_dir=audio_dir, video_dir=video_dir, text_dir=text_dir,
                split_file=split_file, split='val',
                missing_config=missing_config, seed=run_seed
            )
            test_dataset = MOSIDatasetRegression(
                audio_dir=audio_dir, video_dir=video_dir, text_dir=text_dir,
                split_file=split_file, split='test',
                missing_config=missing_config, seed=run_seed
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     shuffle=False, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, collate_fn=collate_fn)
            
            # Initialize model
            model = ClusterImputationModel().to(device)
            
            # STAGE 1: Extract embeddings and perform clustering
            train_embeddings, train_names = extract_all_embeddings(model, train_loader, device)
            val_embeddings, val_names = extract_all_embeddings(model, val_loader, device)
            test_embeddings, test_names = extract_all_embeddings(model, test_loader, device)
            
            # Initialize cluster imputer
            cluster_imputer = ClusterGuidedImputation(
                k_clusters=k_clusters, device=device, seed=run_seed
            )
            
            # Fit clustering on training data
            cluster_imputer.fit(train_embeddings)
            
            # Assign to model
            model.cluster_imputer = cluster_imputer
            
            # STAGE 2: Impute missing modalities
            complete_train_embeddings = impute_all_embeddings(train_embeddings, cluster_imputer, device)
            complete_val_embeddings = impute_all_embeddings(val_embeddings, cluster_imputer, device)
            complete_test_embeddings = impute_all_embeddings(test_embeddings, cluster_imputer, device)
            
            # Get labels
            train_labels = [train_dataset.samples[i]['label'] for i in range(len(train_dataset))]
            val_labels = [val_dataset.samples[i]['label'] for i in range(len(val_dataset))]
            test_labels = [test_dataset.samples[i]['label'] for i in range(len(test_dataset))]
            
            # STAGE 3: Train task head
            train_task(model, complete_train_embeddings, train_labels,
                      complete_val_embeddings, val_labels,
                      num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=device)
            
            # STAGE 4: Evaluate
            print("\n" + "="*80)
            print("TEST EVALUATION")
            print("="*80 + "\n")
            
            mse, mae, rmse, corr, binary_acc, preds, labels = evaluate_model(
                model, complete_test_embeddings, test_labels, device
            )
            f1_micro, f1_macro = calculate_f1_scores(preds, labels)
            sum_run += f1_micro
            
            # Visualise imputation quality (only for configs with missing modalities)
            # print("\nGenerating imputation visualisations...")
            # visualize_imputation(
            #     model=model,
            #     test_embeddings=complete_test_embeddings,
            #     test_dataset=test_dataset,
            #     audio_dir=audio_dir, video_dir=video_dir,
            #     text_dir=text_dir, split_file=split_file,
            #     cluster_imputer=cluster_imputer,
            #     missing_config=missing_config,
            #     run_idx=run_idx,
            #     results_dir=results_dir,
            #     device=device,
            #     batch_size=batch_size,
            # )
            
            print(f"Test Results:")
            print(f"  MSE:               {mse:.4f}")
            print(f"  MAE:               {mae:.4f}")
            print(f"  RMSE:              {rmse:.4f}")
            print(f"  Pearson Corr:      {corr:.4f}")
            print(f"  Binary Accuracy:   {binary_acc:.4f}")
            print(f"  F1 Micro:          {f1_micro:.4f}")
            print(f"  F1 Macro:          {f1_macro:.4f}")
            
            # Save results
            results = {
                "config": missing_config,
                "run": run_idx + 1,
                "seed": run_seed,
                "k_clusters": k_clusters,
                "num_epochs": num_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "test_metrics": {
                    "MSE": float(mse),
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "Pearson_Corr": float(corr),
                    "Binary_Accuracy": float(binary_acc),
                    "F1_Micro": float(f1_micro), 
                    "F1_Macro": float(f1_macro)
                }
            }
            
            results_path = os.path.join(
                results_dir,
                f"cluster-k{k_clusters}-e{num_epochs}-lr{lr}-{missing_config}-{run_idx+1}.json"
            )
            save_results_to_json(results, results_path)
            
            config_results.append(results)
            
            print(f"\n✓ Results saved to {results_path}")
            print(f"\n{'='*80}")

            for r in range(len(recorded_fmicro_scores)):
                print(f"    mean f-micro of {missing_configs[r]}: {recorded_fmicro_scores[r]:.2f}")

            print(f"For {missing_config}:")
            print(f"COMPLETED RUN {run_idx+1}/{num_runs}, average F1 Micro so far: {sum_run/(run_idx+1):.3f}")
            print(f"{'='*80}\n")
        
        recorded_fmicro_scores.append(sum_run / num_runs)
        all_results[missing_config] = config_results
    
    # Save summary
    summary = {
        "num_configs": len(missing_configs),
        "num_runs": num_runs,
        "k_clusters": k_clusters,
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "configs": missing_configs,
        "results": all_results
    }
    
    summary_path = os.path.join(results_dir, f"SUMMARY-cluster-k{k_clusters}.json")
    save_results_to_json(summary, summary_path)
    
    print(f"\n{'#'*80}")
    print("# EXPERIMENT COMPLETE!")
    print(f"{'#'*80}\n")
    print(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    import sys
    
    # Example usage with command line arguments
    if len(sys.argv) > 1:
        configs = sys.argv[1:]
        my_ks = [
            70#, 20, 30, 40, 50, 60, 70, 80, 90, 100
        ]
        for k in my_ks:
            main(missing_configs=configs,
                num_runs=10,
                k_clusters=k,
                num_epochs= 20,
                lr=5e-4,
                batch_size=8
            )
    else:
        # Default: run with a few configurations
        main(
            missing_configs=[
                "100_text_100_audio_100_video",  
                "100_text_20_audio_20_video",  
                "20_text_100_audio_20_video", 
                "20_text_20_audio_100_video",          
                "20_text_100_audio_100_video",
                "100_text_100_audio_20_video",  
                "complex_20_20_20_10_10_10_10", 
                "100_text_20_audio_100_video"
                ],
            num_runs=10,
            k_clusters=10,
            num_epochs= 20,
            lr=5e-4,
            batch_size=8
        )