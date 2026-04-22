"""
PyTorch NER Model for Surface Transport Domain (Task 1)
=======================================================
Architecture: Embedding → BiLSTM → Linear → CRF
Extracts: LOCATION, STATE, VEHICLE, KILLED, INJURED, COLLISION, 
          CAUSE, DATE, ROAD_TYPE, IPC_SECTION, MV_SECTION, etc.

Trains on accident text annotated with BIO tags.
Saves model weights + vocab as .pkl files.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pickle
import os
import re
import json
import shutil
from datetime import datetime
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import functools


def _backup_existing_models(model_dir, prefix=""):
    """Backup existing model files before overwriting.

    Creates a timestamped folder inside <model_dir>/backups/ and copies
    all .pt and .pkl files there so previous training runs are preserved.
    """
    model_files = [f for f in os.listdir(model_dir)
                   if f.endswith(('.pt', '.pkl')) and (not prefix or f.startswith(prefix))]
    if not model_files:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(model_dir, "backups", ts)
    os.makedirs(backup_dir, exist_ok=True)
    for f in model_files:
        src = os.path.join(model_dir, f)
        shutil.copy2(src, os.path.join(backup_dir, f))
    print(f"  ⮡ Backed up {len(model_files)} existing model files → {backup_dir}")


# ═══════════════════════════════════════════════════════════════
# FAST GAZETTEER INDEX
# Rebuilt once after GAZETTEERS is mutated (i.e. after Excel
# locations/vehicles are injected).  Replaces the O(terms × tokens)
# triple loop with a single O(tokens) pass via a token-prefix trie.
# ═══════════════════════════════════════════════════════════════

# Module-level cache — call build_gazetteer_index() to refresh.
_GAZE_INDEX: dict = {}   # token_tuple → entity_type


def build_gazetteer_index() -> None:
    """
    Build a flat dict mapping lowercased token-tuples → entity_type
    from the current GAZETTEERS dict.  Call this once after all
    Excel locations/vehicles have been injected.
    """
    global _GAZE_INDEX
    idx: dict = {}
    for entity_type, terms in GAZETTEERS.items():
        for term in terms:
            key = tuple(re.findall(r'\b\w+\b|[^\w\s]', term.lower()))
            if key:
                idx[key] = entity_type
    _GAZE_INDEX = idx


def _gaze_match_fast(tokens: list, tags: list) -> None:
    """
    Single O(N) pass over tokens using _GAZE_INDEX.
    For every position i, tries all prefix lengths that exist in index.
    Worst-case length per position = max term length (≤ ~6 tokens).
    """
    if not _GAZE_INDEX:
        return
    lower = [t.lower() for t in tokens]
    n = len(lower)
    max_len = max(len(k) for k in _GAZE_INDEX) if _GAZE_INDEX else 1

    i = 0
    while i < n:
        best_len = 0
        best_type = None
        for length in range(1, min(max_len, n - i) + 1):
            key = tuple(lower[i:i + length])
            if key in _GAZE_INDEX:
                best_len = length
                best_type = _GAZE_INDEX[key]
        if best_len > 0 and all(tags[i + j] == "O" for j in range(best_len)):
            tags[i] = f"B-{best_type}"
            for j in range(1, best_len):
                tags[i + j] = f"I-{best_type}"
            i += best_len
        else:
            i += 1


# ═══════════════════════════════════════════════════════════════
# 1. BIO TAG DEFINITIONS
# ═══════════════════════════════════════════════════════════════

ENTITY_LABELS = [
    "O",
    "B-LOCATION", "I-LOCATION",
    "B-STATE", "I-STATE",
    "B-VEHICLE", "I-VEHICLE",
    "B-NUM_KILLED", "I-NUM_KILLED",
    "B-NUM_INJURED", "I-NUM_INJURED",
    "B-COLLISION", "I-COLLISION",
    "B-CAUSE", "I-CAUSE",
    "B-DATE", "I-DATE",
    "B-TIME", "I-TIME",
    "B-ROAD_TYPE", "I-ROAD_TYPE",
    "B-WEATHER", "I-WEATHER",
    "B-IPC_SECTION", "I-IPC_SECTION",
    "B-MV_SECTION", "I-MV_SECTION",
    "B-REG_NUMBER", "I-REG_NUMBER",
    "B-VICTIM_TYPE", "I-VICTIM_TYPE",
    "B-SEVERITY", "I-SEVERITY",
]

LABEL2IDX = {l: i for i, l in enumerate(ENTITY_LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}
NUM_LABELS = len(ENTITY_LABELS)

# Start/Stop tags for CRF
START_TAG = "<START>"
STOP_TAG = "<STOP>"
LABEL2IDX[START_TAG] = NUM_LABELS
LABEL2IDX[STOP_TAG] = NUM_LABELS + 1
IDX2LABEL[NUM_LABELS] = START_TAG
IDX2LABEL[NUM_LABELS + 1] = STOP_TAG
TAGSET_SIZE = NUM_LABELS + 2


# ═══════════════════════════════════════════════════════════════
# 2. VOCABULARY BUILDER
# ═══════════════════════════════════════════════════════════════

class Vocabulary:
    """Word-level vocabulary with special tokens."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, min_freq=1):
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build(self, sentences: list):
        """Build vocab from list of tokenized sentences."""
        for tokens in sentences:
            self.word_freq.update([t.lower() for t in tokens])
        idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        return self

    def encode(self, tokens: list) -> list:
        return [self.word2idx.get(t.lower(), self.word2idx[self.UNK]) for t in tokens]

    def decode(self, indices: list) -> list:
        return [self.idx2word.get(i, self.UNK) for i in indices]

    @property
    def size(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word,
                         'word_freq': self.word_freq, 'min_freq': self.min_freq}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_freq = data.get('word_freq', Counter())
        self.min_freq = data.get('min_freq', 1)
        return self


# ═══════════════════════════════════════════════════════════════
# 3. CRF LAYER
# ═══════════════════════════════════════════════════════════════

class CRF(nn.Module):
    """Conditional Random Field layer for sequence labeling."""

    def __init__(self, tagset_size):
        super().__init__()
        self.tagset_size = tagset_size
        # transitions[i][j] = score of transitioning FROM tag j TO tag i
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        # No transition TO start, no transition FROM stop
        self.transitions.data[LABEL2IDX[START_TAG], :] = -10000
        self.transitions.data[:, LABEL2IDX[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        """Forward algorithm to compute partition function."""
        init_alphas = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
        init_alphas[0][LABEL2IDX[START_TAG]] = 0.0
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[LABEL2IDX[STOP_TAG]]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags):
        """Score a tagged sequence."""
        score = torch.zeros(1, device=feats.device)
        tags = torch.cat([torch.tensor([LABEL2IDX[START_TAG]], device=feats.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[LABEL2IDX[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """Find the best tag sequence via Viterbi."""
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
        init_vvars[0][LABEL2IDX[START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = next_tag_var.argmax(dim=1).item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[LABEL2IDX[STOP_TAG]]
        best_tag_id = terminal_var.argmax(dim=1).item()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.pop()  # Remove START_TAG
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags):
        """Compute negative log likelihood loss."""
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, feats):
        """Viterbi decode for inference."""
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq


# ═══════════════════════════════════════════════════════════════
# 4. BiLSTM-CRF NER MODEL
# ═══════════════════════════════════════════════════════════════

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF model for Named Entity Recognition.

    Architecture:
        Word Embedding (+ optional char features)
        → Bidirectional LSTM
        → Linear projection to tag space
        → CRF decoding

    Parameters (with default config):
        Embedding: vocab_size × 128 = ~128K params
        BiLSTM: 128→256 (2 layers) = ~400K params
        Linear: 256→34 = ~8.7K params
        CRF: 34×34 = ~1.2K params
        Total: ~538K+ parameters (well above 1000)
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, TAGSET_SIZE)
        self.crf = CRF(TAGSET_SIZE)

    def _get_lstm_features(self, sentence):
        """Get emission scores from BiLSTM."""
        embeds = self.dropout(self.embedding(sentence))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):
        """Predict best tag sequence (inference)."""
        lstm_feats = self._get_lstm_features(sentence)
        # CRF works on single sequence (not batched)
        if lstm_feats.dim() == 3:
            lstm_feats = lstm_feats.squeeze(0)
        score, tag_seq = self.crf(lstm_feats)
        return score, tag_seq

    def loss(self, sentence, tags):
        """Compute CRF loss for training."""
        lstm_feats = self._get_lstm_features(sentence)
        if lstm_feats.dim() == 3:
            lstm_feats = lstm_feats.squeeze(0)
        if tags.dim() == 2:
            tags = tags.squeeze(0)
        return self.crf.neg_log_likelihood(lstm_feats, tags)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# 5. AUTOMATIC ANNOTATION ENGINE
# ═══════════════════════════════════════════════════════════════

# Domain gazetteers for auto-annotation
GAZETTEERS = {
    "LOCATION": [
        # Base cities — Excel data will add hundreds more during training
        "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata",
        "pune", "ahmedabad", "jaipur", "lucknow", "surat", "nagpur",
        "indore", "bhopal", "patna", "vadodara", "ludhiana", "agra",
        "nashik", "jodhpur", "varanasi", "coimbatore", "chandigarh",
    ],
    "STATE": [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal", "Delhi", "Jammu and Kashmir",
    ],
    "VEHICLE": [
        "car", "truck", "bus", "auto rickshaw", "motorcycle", "bike", "scooter",
        "two wheeler", "lorry", "tempo", "tractor", "van", "jeep", "taxi", "SUV",
        "tanker", "trailer", "ambulance", "bicycle", "cycle", "dumper", "tipper",
        "e-rickshaw", "mini bus", "school bus", "pick up",
    ],
    "COLLISION": [
        "head on", "head-on", "rear end", "rear-end", "hit from back",
        "hit from side", "side swipe", "overturn", "overturned", "rollover",
        "run off road", "hit and run", "hit-and-run", "pile up", "pile-up",
        "collision", "rammed", "skidding",
    ],
    "CAUSE": [
        "overspeeding", "over speeding", "speeding", "rash driving",
        "drunk driving", "drunken driving", "negligence", "wrong side",
        "red light", "signal jumping", "dangerous overtaking",
        "mobile phone", "brake failure", "tyre burst", "overloading",
        "poor visibility", "fog", "pothole",
    ],
    "ROAD_TYPE": [
        "national highway", "state highway", "expressway", "district road",
        "village road", "flyover", "bridge", "toll plaza",
    ],
    "WEATHER": [
        "fog", "rain", "heavy rain", "storm", "flooding", "hail", "snow",
        "mist", "poor visibility", "dust storm",
    ],
    "SEVERITY": ["fatal", "grievous", "simple", "minor"],
    "VICTIM_TYPE": ["pedestrian", "cyclist", "driver", "passenger", "pillion rider"],
}

WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}


def tokenize(text: str) -> list:
    """Simple word tokenizer preserving punctuation."""
    return re.findall(r'\b\w+\b|[^\w\s]', text)


def auto_annotate(text: str) -> list:
    """
    Automatically create BIO-tagged training data from raw text
    using gazetteer matching and regex patterns.

    Returns: list of (token, BIO_tag) tuples
    """
    tokens = tokenize(text)
    tags = ["O"] * len(tokens)
    text_lower = text.lower()

    # Helper: tag a span of tokens
    def tag_span(start_idx, length, entity_type):
        if start_idx + length > len(tokens):
            return
        tags[start_idx] = f"B-{entity_type}"
        for j in range(1, length):
            tags[start_idx + j] = f"I-{entity_type}"

    # 1. Gazetteer matching — fast trie-based single pass
    _gaze_match_fast(tokens, tags)

    # 2. Number patterns for killed/injured
    for i, token in enumerate(tokens):
        t_lower = token.lower()

        # "5 killed", "3 injured"
        if token.isdigit() and i + 1 < len(tokens):
            next_t = tokens[i + 1].lower()
            if next_t in ("killed", "died", "dead", "fatalities"):
                if tags[i] == "O":
                    tags[i] = "B-NUM_KILLED"
            elif next_t in ("injured", "hurt", "wounded", "hospitalised", "hospitalized"):
                if tags[i] == "O":
                    tags[i] = "B-NUM_INJURED"

        # "Five persons killed"
        if t_lower in WORD_NUMBERS and i + 1 < len(tokens):
            next_words = " ".join(tokens[i + 1:i + 3]).lower()
            if any(w in next_words for w in ["killed", "died", "dead"]):
                if tags[i] == "O":
                    tags[i] = "B-NUM_KILLED"
            elif any(w in next_words for w in ["injured", "hurt"]):
                if tags[i] == "O":
                    tags[i] = "B-NUM_INJURED"

        # "killing 5", "injuring 3"
        if t_lower in ("killing", "injuring") and i + 1 < len(tokens):
            if tokens[i + 1].isdigit() and tags[i + 1] == "O":
                etype = "NUM_KILLED" if t_lower == "killing" else "NUM_INJURED"
                tags[i + 1] = f"B-{etype}"

    # 3. Date patterns
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, token in enumerate(tokens):
        t_lower = token.lower()
        if t_lower in months and tags[i] == "O":
            tags[i] = "B-DATE"
            # Check for day number before/after
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                tags[i + 1] = "I-DATE"
            if i > 0 and tokens[i - 1].isdigit() and tags[i - 1] == "O":
                tags[i - 1] = "B-DATE"
                tags[i] = "I-DATE"
        if t_lower in days and tags[i] == "O":
            tags[i] = "B-DATE"

    # 4. NH/SH road numbers
    for i, token in enumerate(tokens):
        if token.upper() in ("NH", "SH") and tags[i] == "O":
            tags[i] = "B-ROAD_TYPE"
            # Tag the number after it
            if i + 1 < len(tokens):
                next_t = tokens[i + 1]
                if next_t.isdigit() or re.match(r'^\d+[A-Za-z]?$', next_t):
                    tags[i + 1] = "I-ROAD_TYPE"

    # 5. IPC sections
    ipc_sections = ["279", "304A", "337", "338", "302"]
    for i, token in enumerate(tokens):
        if token.lower() in ("section", "sec") and i + 1 < len(tokens):
            if tokens[i + 1] in ipc_sections:
                if tags[i + 1] == "O":
                    tags[i + 1] = "B-IPC_SECTION"

    # 6. Registration numbers (e.g., RJ 14 GA 1234)
    for i in range(len(tokens) - 3):
        if (re.match(r'^[A-Z]{2}$', tokens[i]) and
                tokens[i + 1].isdigit() and
                re.match(r'^[A-Z]{1,3}$', tokens[i + 2]) and
                re.match(r'^\d{4}$', tokens[i + 3])):
            if all(tags[i + j] == "O" for j in range(4)):
                tag_span(i, 4, "REG_NUMBER")

    # Validate: only use known tags
    for i in range(len(tags)):
        if tags[i] not in LABEL2IDX:
            tags[i] = "O"

    return list(zip(tokens, tags))


def _annotate_one(text: str):
    """Worker for parallel annotation (module-level so pickle works)."""
    if len(text.strip()) < 20:
        return None
    annotated = auto_annotate(text)
    if not annotated:
        return None
    tokens = [a[0] for a in annotated]
    tags   = [a[1] for a in annotated]
    if any(t != "O" for t in tags):
        return (tokens, tags)
    return None


def create_training_data_from_texts(texts: list) -> list:
    """
    Create BIO-annotated training data from a list of raw texts.
    Uses multiprocessing to parallelize annotation across CPU cores.
    Returns list of (tokens_list, tags_list) pairs.
    """
    # Rebuild the gazetteer index once before annotation starts
    build_gazetteer_index()
    print(f"  Gazetteer index built: {len(_GAZE_INDEX):,} term entries")

    workers = max(1, cpu_count() - 1)
    chunk_size = max(1, len(texts) // (workers * 4))
    print(f"  Annotating {len(texts):,} texts using {workers} workers "
          f"(chunk_size={chunk_size})...")

    with Pool(processes=workers) as pool:
        results = pool.map(_annotate_one, texts, chunksize=chunk_size)

    training_data = [r for r in results if r is not None]
    print(f"  -> {len(training_data):,} annotated samples retained")
    return training_data


# ═══════════════════════════════════════════════════════════════
# 6. DATASET
# ═══════════════════════════════════════════════════════════════

class NERDataset(Dataset):
    """Dataset for NER training."""

    def __init__(self, data, vocab):
        """
        Args:
            data: list of (tokens_list, tags_list) pairs
            vocab: Vocabulary object
        """
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        token_ids = torch.tensor(self.vocab.encode(tokens), dtype=torch.long)
        tag_ids = torch.tensor([LABEL2IDX.get(t, 0) for t in tags], dtype=torch.long)
        return token_ids, tag_ids, len(tokens)


def collate_fn(batch):
    """Pad sequences in a batch."""
    tokens, tags, lengths = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    return tokens_padded, tags_padded, torch.tensor(lengths)


# ═══════════════════════════════════════════════════════════════
# 7. TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

def train_ner(training_data, vocab, model_dir="models",
              epochs=50, lr=0.005, hidden_dim=128, embedding_dim=128,
              max_samples=3000):
    """
    Train the BiLSTM-CRF NER model.

    Args:
        training_data: list of (tokens, tags) pairs
        vocab: Vocabulary object
        model_dir: where to save model files
        max_samples: cap training samples per epoch for speed (default 3000)
    """
    try:
        from tqdm import tqdm
        USE_TQDM = True
    except ImportError:
        USE_TQDM = False

    os.makedirs(model_dir, exist_ok=True)

    # ── KEY FIX: CRF Python loops are SLOWER on CUDA than CPU ──
    # The CRF _forward_alg is a pure Python loop that launches one tiny
    # CUDA kernel per token — massive overhead. CPU is faster for this.
    # LSTM runs on CUDA (fast), CRF runs on CPU (fast). Best of both.
    lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crf_device  = torch.device("cpu")  # CRF always on CPU

    print(f"\n{'='*60}")
    print(f"  TRAINING BiLSTM-CRF NER MODEL")
    print(f"  LSTM device : {lstm_device}")
    print(f"  CRF  device : {crf_device}  ← Python loops, faster on CPU")
    print(f"  Training samples: {len(training_data)} (capped at {max_samples}/epoch)")
    print(f"  Vocabulary size : {vocab.size}")
    print(f"  Tag set size    : {TAGSET_SIZE}")
    print(f"{'='*60}\n")

    model = BiLSTM_CRF(
        vocab_size=vocab.size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(lstm_device)
    # Move ONLY the CRF layer to CPU — its Python loops are faster there.
    # Embedding + LSTM stay on GPU for fast forward pass.
    model.crf = model.crf.to(crf_device)

    total_params = model.count_parameters()
    print(f"  Total trainable parameters: {total_params:,}")
    assert total_params >= 1000, f"Only {total_params} params, need ≥1000"
    print(f"  ✓ Parameter requirement met (≥ 1,000)\n")

    # Estimate and warn about training time upfront
    samples_per_epoch = min(len(training_data), max_samples)
    print(f"  ⏱  Estimated time: ~{samples_per_epoch * epochs // 6000 + 1} min "
          f"({samples_per_epoch} samples × {epochs} epochs)\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss  = 0
        count       = 0

        # Shuffle and cap samples per epoch for speed
        epoch_data = list(training_data)
        np.random.shuffle(epoch_data)
        epoch_data = epoch_data[:max_samples]

        # Progress bar per epoch
        if USE_TQDM:
            iterator = tqdm(epoch_data,
                            desc=f"Epoch {epoch:3d}/{epochs}",
                            unit="seq", leave=False,
                            bar_format="{l_bar}{bar:30}{r_bar}")
        else:
            iterator = epoch_data
            # Fallback: print every 500 samples within epoch 1
            if epoch == 1:
                print(f"  (install tqdm for progress bars: pip install tqdm)")

        for tokens, tags in iterator:
            # LSTM forward on GPU (fast)
            token_ids = torch.tensor(
                vocab.encode(tokens), dtype=torch.long
            ).unsqueeze(0).to(lstm_device)

            tag_ids = torch.tensor(
                [LABEL2IDX.get(t, 0) for t in tags], dtype=torch.long
            ).to(crf_device)  # CRF tags stay on CPU

            optimizer.zero_grad()

            # Get LSTM features on GPU, then move to CPU for CRF
            lstm_feats = model._get_lstm_features(token_ids)
            if lstm_feats.dim() == 3:
                lstm_feats = lstm_feats.squeeze(0)
            lstm_feats_cpu = lstm_feats.to(crf_device)  # Move to CPU for CRF

            loss = model.crf.neg_log_likelihood(lstm_feats_cpu, tag_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            count      += 1

            if USE_TQDM:
                iterator.set_postfix(loss=f"{loss.item():.3f}")
            elif epoch == 1 and count % 500 == 0:
                print(f"    [Epoch 1] {count}/{len(epoch_data)} samples done, "
                      f"loss so far: {total_loss/count:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)

        # Always print epoch summary (not just every 5)
        print(f"  Epoch {epoch:3d}/{epochs} │ Loss: {avg_loss:.4f} │ "
              f"LR: {scheduler.get_last_lr()[0]:.6f} │ "
              f"Samples: {count}")

    # Backup existing models before overwriting
    _backup_existing_models(model_dir, prefix="ner")

    # Save model
    model_path = os.path.join(model_dir, "ner_bilstm_crf.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'vocab_size': vocab.size,
        'loss_history': loss_history,
        'total_params': total_params,
    }, model_path)
    print(f"\n  ✓ Model saved: {model_path} ({os.path.getsize(model_path):,} bytes)")

    # Save vocab
    vocab_path = os.path.join(model_dir, "ner_vocab.pkl")
    vocab.save(vocab_path)
    print(f"  ✓ Vocab saved: {vocab_path} ({os.path.getsize(vocab_path):,} bytes)")

    # Save label mapping
    labels_path = os.path.join(model_dir, "ner_labels.pkl")
    with open(labels_path, 'wb') as f:
        pickle.dump({'label2idx': LABEL2IDX, 'idx2label': IDX2LABEL}, f)
    print(f"  ✓ Labels saved: {labels_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — Final Loss: {loss_history[-1]:.4f}")
    print(f"  Parameters: {total_params:,}")
    print(f"{'='*60}\n")

    return model


# ═══════════════════════════════════════════════════════════════
# 8. INFERENCE
# ═══════════════════════════════════════════════════════════════

class NERPredictor:
    """Load trained model and predict entities from text."""

    def __init__(self, model_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model = None
        self.vocab = None

    def load(self):
        """Load model, vocab, and labels."""
        # Load vocab
        vocab_path = os.path.join(self.model_dir, "ner_vocab.pkl")
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)

        # Load model
        model_path = os.path.join(self.model_dir, "ner_bilstm_crf.pt")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = BiLSTM_CRF(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"NER model loaded ({checkpoint['total_params']:,} parameters)")

    def predict(self, text: str) -> list:
        """Predict NER tags for input text."""
        tokens = tokenize(text)
        if not tokens:
            return []

        token_ids = torch.tensor(
            self.vocab.encode(tokens), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score, tag_ids = self.model(token_ids)

        predicted_tags = [IDX2LABEL.get(t, "O") for t in tag_ids]

        # Truncate/pad to match token length
        predicted_tags = predicted_tags[:len(tokens)]
        while len(predicted_tags) < len(tokens):
            predicted_tags.append("O")

        return list(zip(tokens, predicted_tags))

    def extract_entities(self, text: str) -> dict:
        """Extract entities grouped by type."""
        predictions = self.predict(text)
        entities = defaultdict(list)
        current_entity = None
        current_tokens = []

        for token, tag in predictions:
            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities[current_entity].append(" ".join(current_tokens))
                current_entity = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_tokens.append(token)
            else:
                if current_entity:
                    entities[current_entity].append(" ".join(current_tokens))
                current_entity = None
                current_tokens = []

        # Save last entity
        if current_entity:
            entities[current_entity].append(" ".join(current_tokens))

        return dict(entities)

    def extract_to_far_format(self, text: str) -> dict:
        """Map extracted entities to FAR form fields."""
        entities = self.extract_entities(text)
        word_map = {"one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8",
                    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}

        def get_first(key):
            vals = entities.get(key, [])
            if vals:
                v = vals[0]
                return word_map.get(v.lower(), v)
            return ""

        killed = get_first("NUM_KILLED")
        far = {
            "date_of_accident": get_first("DATE"),
            "time_of_accident": get_first("TIME"),
            "place_of_accident": get_first("LOCATION"),
            "state": get_first("STATE"),
            "nature_of_accident": "Fatal" if killed and killed not in ("0", "") else get_first("SEVERITY"),
            "vehicle_1": entities.get("VEHICLE", [""])[0] if entities.get("VEHICLE") else "",
            "vehicle_2": entities.get("VEHICLE", ["", ""])[1] if len(entities.get("VEHICLE", [])) > 1 else "",
            "registration_number": get_first("REG_NUMBER"),
            "collision_type": get_first("COLLISION"),
            "number_of_fatalities": killed or "0",
            "number_of_injured": get_first("NUM_INJURED") or "0",
            "road_type": get_first("ROAD_TYPE"),
            "weather_condition": get_first("WEATHER"),
            "cause_of_accident": ", ".join(entities.get("CAUSE", [])),
            "victim_type": ", ".join(entities.get("VICTIM_TYPE", [])),
            "ipc_sections": ", ".join(entities.get("IPC_SECTION", [])),
            "mv_act_sections": ", ".join(entities.get("MV_SECTION", [])),
        }
        return far


# ═══════════════════════════════════════════════════════════════
# 9. MAIN — TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing auto-annotation...")
    test = ("Five persons died and three dozen were injured in a head-on "
            "collision of a passenger bus and a truck near Mathania in "
            "Jodhpur on Friday afternoon in Rajasthan. The accident was "
            "caused by overspeeding on State Highway 61. Police registered "
            "a case under Section 304A of the IPC.")

    annotated = auto_annotate(test)
    print("\nAnnotated tokens:")
    for tok, tag in annotated:
        if tag != "O":
            print(f"  {tok:20s} → {tag}")

    print(f"\nTotal tokens: {len(annotated)}")
    print(f"Entity tokens: {sum(1 for _, t in annotated if t != 'O')}")