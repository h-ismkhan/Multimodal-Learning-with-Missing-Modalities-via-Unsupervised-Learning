import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import transforms

# Suppress PIL decompression bomb warnings
Image.MAX_IMAGE_PIXELS = 200000000

def analyze_dataset(dataset_path, split_path):
    """Analyze the MM-IMDb dataset - discovers genres dynamically"""
    print("\n" + "="*80)
    print("ANALYZING MM-IMDB DATASET (DYNAMIC GENRE DISCOVERY)")
    print("="*80)
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    all_genres = set()
    sample_info = {}
    genre_counts = Counter()
    
    for split_name in ['train', 'dev', 'test']:
        for sample_id in splits[split_name]:
            json_path = os.path.join(dataset_path, 'dataset', f'{sample_id}.json')
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    genres = data.get('genres', [])
                    plot = data.get('plot', [''])[0] if isinstance(data.get('plot'), list) else data.get('plot', '')
                    
                    all_genres.update(genres)
                    genre_counts.update(genres)
                    sample_info[sample_id] = {
                        'genres': genres,
                        'plot': plot,
                        'split': split_name
                    }
    
    genre_list = sorted(list(all_genres))
    genre_to_idx = {genre: idx for idx, genre in enumerate(genre_list)}
    
    print(f"\nDataset Statistics:")
    print(f"  Total unique genres found: {len(genre_list)}")
    print(f"  Genres (sorted): {genre_list}")
    print(f"\n  Genre Distribution (top 10 most common):")
    for genre, count in genre_counts.most_common(10):
        print(f"    {genre:<15s}: {count:>4d} samples")
    print(f"\n  Split sizes:")
    print(f"    Train: {len(splits['train'])}")
    print(f"    Dev:   {len(splits['dev'])}")
    print(f"    Test:  {len(splits['test'])}")
    print("="*80 + "\n")
    
    return splits, genre_list, genre_to_idx, sample_info


class MMIMDbDatasetCLIP(Dataset):
    """
    MM-IMDb Dataset for CLIP encoders
    Returns raw images and text strings (not tokenized)
    """
    def __init__(self, root_dir, split_file, split='train',
                 genre_list=None, genre_to_idx=None,
                 missing_config='100_image_80_text',
                 img_size=224, seed=42):
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, 'dataset')
        self.split = split
        self.genre_list = genre_list
        self.genre_to_idx = genre_to_idx
        self.missing_config = missing_config
        self.img_size = img_size
        self.seed = seed
        
        # Load split information
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.sample_ids = splits[split]
        
        # Parse missing configuration
        self.parse_missing_config()
        
        # Pre-compute deterministic modality availability
        self.modality_availability = self._precompute_modality_availability()
        
        # Image transforms for CLIP (standard ImageNet normalization)
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self._print_dataset_info()
    
    def parse_missing_config(self):
        """Parse missing configuration string"""
        if self.missing_config == '100_image_100_text':
            self.image_ratio = 1.0
            self.text_ratio = 1.0
            self.complex_mode = False
        elif self.missing_config.startswith('100_image_'):
            self.image_ratio = 1.0
            parts = self.missing_config.split('_')
            text_pct = int(parts[2])
            self.text_ratio = text_pct / 100.0
            self.complex_mode = False
        elif self.missing_config.endswith('_100_text'):
            img_pct = int(self.missing_config.split('_')[0])
            self.image_ratio = img_pct / 100.0
            self.text_ratio = 1.0
            self.complex_mode = False
        elif self.missing_config.startswith('complex_'):
            parts = self.missing_config.split('_')
            self.alpha = int(parts[1]) / 100.0  # both
            self.beta = int(parts[2]) / 100.0   # image only
            self.gamma = int(parts[3]) / 100.0  # text only
            
            total = self.alpha + self.beta + self.gamma
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Complex ratios must sum to 1.0, got {total}")
            
            self.complex_mode = True
            self.image_ratio = None
            self.text_ratio = None
        else:
            raise ValueError(f"Unknown missing config: {self.missing_config}")
    
    def _precompute_modality_availability(self):
        """Pre-compute modality availability deterministically"""
        rng = np.random.RandomState(self.seed)
        availability = {}
        
        if self.complex_mode:
            for idx in range(len(self.sample_ids)):
                rand_val = rng.random()
                if rand_val < self.alpha:
                    availability[idx] = (True, True)
                elif rand_val < self.alpha + self.beta:
                    availability[idx] = (True, False)
                else:
                    availability[idx] = (False, True)
        else:
            for idx in range(len(self.sample_ids)):
                has_image = rng.random() < self.image_ratio
                has_text = rng.random() < self.text_ratio
                availability[idx] = (has_image, has_text)
        
        return availability
    
    def _print_dataset_info(self):
        """Print dataset statistics"""
        both_count = sum(1 for (img, txt) in self.modality_availability.values() if img and txt)
        img_only_count = sum(1 for (img, txt) in self.modality_availability.values() if img and not txt)
        text_only_count = sum(1 for (img, txt) in self.modality_availability.values() if not img and txt)
        
        total = len(self.sample_ids)
        
        print(f"\n{'='*80}")
        print(f"MM-IMDb Dataset (CLIP): {self.split} split")
        print(f"  Config: {self.missing_config} | Seed: {self.seed}")
        print(f"  Total: {total} | Both: {both_count} ({100*both_count/total:.1f}%) | "
              f"Img: {img_only_count} ({100*img_only_count/total:.1f}%) | "
              f"Text: {text_only_count} ({100*text_only_count/total:.1f}%)")
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def load_sample_data(self, sample_id):
        """Load image, text, and labels"""
        json_path = os.path.join(self.dataset_dir, f"{sample_id}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Get plot text
        plot = metadata.get('plot', [''])[0] if metadata.get('plot') else ''
        if isinstance(plot, list):
            plot = ' '.join(plot)
        
        # Get genres
        genres = metadata.get('genres', [])
        
        # Multi-label vector
        label = torch.zeros(len(self.genre_list))
        for genre in genres:
            if genre in self.genre_to_idx:
                label[self.genre_to_idx[genre]] = 1.0
        
        # Load image
        img_path = os.path.join(self.dataset_dir, f"{sample_id}.jpeg")
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            # Fallback to zeros if image loading fails
            image = torch.zeros(3, self.img_size, self.img_size)
        
        return image, plot, label
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image, text, label = self.load_sample_data(sample_id)
        
        # Get pre-computed modality availability
        has_image, has_text = self.modality_availability[idx]
        
        # Apply missing modality masking
        if not has_image:
            image = torch.zeros_like(image)
        if not has_text:
            text = ""  # Empty string for missing text
        
        return {
            'image': image,
            'text': text,
            'label': label,
            'has_image': has_image,
            'has_text': has_text
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch]),
        'has_image': [item['has_image'] for item in batch],
        'has_text': [item['has_text'] for item in batch]
    }