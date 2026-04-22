"""
PyTorch QA Model for Surface Transport Domain (Task 2)
======================================================
Architecture: Word Embedding → BiLSTM → Attention → Passage Scoring
Retrieves relevant passages from corpus to answer regulatory questions.

Also includes the FormGenerator (same as before, no neural component needed).

Parameters: ~200K+ (well above 1000 minimum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import re
import shutil
from datetime import datetime
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# 1. VOCABULARY (shared with NER)
# ═══════════════════════════════════════════════════════════════

class QAVocabulary:
    """Word-level vocabulary for QA."""
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_vocab=15000, min_freq=1):
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}
        self.max_vocab = max_vocab
        self.min_freq = min_freq

    def build(self, texts: list):
        """Build vocab from list of text strings."""
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
        idx = len(self.word2idx)
        for word, freq in counter.most_common(self.max_vocab):
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        return self

    def encode(self, text: str, max_len: int = 64) -> list:
        tokens = self._tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        # Pad to max_len
        ids += [0] * (max_len - len(ids))
        return ids

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    @property
    def size(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        return self


# ═══════════════════════════════════════════════════════════════
# 2. NEURAL PASSAGE ENCODER
# ═══════════════════════════════════════════════════════════════

class PassageEncoder(nn.Module):
    """
    Encodes text passages into fixed-size vectors using BiLSTM + Attention.

    Architecture:
        Word Embedding (vocab × 96)
        → BiLSTM (96 → 128, 2 layers)
        → Self-Attention pooling
        → Dense projection (128 → 64)

    Used for both queries and passages — shared encoder (Siamese style).
    """

    def __init__(self, vocab_size, embed_dim=96, hidden_dim=64,
                 output_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            (batch, output_dim) passage embeddings
        """
        mask = (x != 0).float()  # (batch, seq_len)
        embeds = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embeds)  # (batch, seq_len, hidden*2)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)

        projected = self.projection(context)
        return self.layer_norm(projected)


class NeuralQAModel(nn.Module):
    """
    Neural QA Retriever using dual encoder (Siamese) architecture.

    Both query and passage go through the same encoder.
    Score = cosine similarity between query and passage vectors.

    Parameters breakdown:
        Embedding: 15000 × 96 = 1,440,000
        BiLSTM: ~200K
        Attention + Projection: ~10K
        Total: ~1.65M parameters
    """

    def __init__(self, vocab_size, embed_dim=96, hidden_dim=64,
                 output_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = PassageEncoder(
            vocab_size, embed_dim, hidden_dim,
            output_dim, num_layers, dropout
        )

    def encode_query(self, query_ids):
        return self.encoder(query_ids)

    def encode_passage(self, passage_ids):
        return self.encoder(passage_ids)

    def forward(self, query_ids, passage_ids):
        """Compute similarity between query and passages."""
        q_vec = self.encode_query(query_ids)  # (batch, dim)
        p_vec = self.encode_passage(passage_ids)  # (batch, dim)
        # Cosine similarity
        q_norm = F.normalize(q_vec, p=2, dim=1)
        p_norm = F.normalize(p_vec, p=2, dim=1)
        return (q_norm * p_norm).sum(dim=1)  # (batch,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def create_qa_training_pairs(chunks: list, num_negatives=3):
    """
    Create (query, positive_passage, negative_passages) training pairs.

    Strategy: use first sentence of each chunk as a pseudo-query,
    the full chunk as the positive passage, and random other chunks
    as negatives (contrastive learning).
    """
    pairs = []
    for i, chunk in enumerate(chunks):
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        if not sentences or len(sentences) < 1:
            continue

        # First sentence (or first 15 words) as query
        query = " ".join(sentences[0].split()[:15])
        positive = chunk

        # Random negatives
        neg_indices = [j for j in range(len(chunks)) if j != i]
        if len(neg_indices) < num_negatives:
            neg_indices = neg_indices * (num_negatives // max(len(neg_indices), 1) + 1)
        np.random.shuffle(neg_indices)
        negatives = [chunks[j] for j in neg_indices[:num_negatives]]

        pairs.append({
            "query": query,
            "positive": positive,
            "negatives": negatives,
        })

    return pairs


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

def train_qa(corpus_chunks: list, chunk_sources: list, vocab: QAVocabulary,
             model_dir="models", epochs=30, lr=0.001, max_seq_len=64):
    """Train the neural QA retriever."""
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  TRAINING NEURAL QA RETRIEVER")
    print(f"  Device: {device}")
    print(f"  Corpus chunks: {len(corpus_chunks)}")
    print(f"  Vocabulary size: {vocab.size}")
    print(f"{'='*60}\n")

    model = NeuralQAModel(
        vocab_size=vocab.size,
        embed_dim=96, hidden_dim=64, output_dim=64,
    ).to(device)

    total_params = model.count_parameters()
    print(f"  Total trainable parameters: {total_params:,}")
    assert total_params >= 1000, f"Only {total_params} params, need ≥1000"
    print(f"  ✓ Parameter requirement met (≥ 1,000)\n")

    # Create training pairs
    print("  Creating contrastive training pairs...")
    training_pairs = create_qa_training_pairs(corpus_chunks, num_negatives=3)
    print(f"  -> {len(training_pairs)} training pairs\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        np.random.shuffle(training_pairs)

        for pair in training_pairs:
            query_ids = torch.tensor(
                [vocab.encode(pair["query"], max_seq_len)],
                dtype=torch.long, device=device
            )
            pos_ids = torch.tensor(
                [vocab.encode(pair["positive"], max_seq_len)],
                dtype=torch.long, device=device
            )

            # Positive score
            pos_score = model(query_ids, pos_ids).squeeze()

            # Negative scores
            neg_scores = []
            for neg in pair["negatives"]:
                neg_ids = torch.tensor(
                    [vocab.encode(neg, max_seq_len)],
                    dtype=torch.long, device=device
                )
                neg_scores.append(model(query_ids, neg_ids).squeeze())

            # Contrastive loss: positive should score higher than negatives
            # Using margin-based triplet loss
            loss = torch.tensor(0.0, device=device)
            margin = 0.2
            for neg_score in neg_scores:
                loss += F.relu(margin - pos_score + neg_score)
            loss = loss / max(len(neg_scores), 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(training_pairs), 1)
        loss_history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} │ Loss: {avg_loss:.4f} │ "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── Pre-compute all passage embeddings ──
    print("\n  Pre-computing passage embeddings...")
    model.eval()
    all_passage_vecs = []
    with torch.no_grad():
        for chunk in corpus_chunks:
            ids = torch.tensor(
                [vocab.encode(chunk, max_seq_len)],
                dtype=torch.long, device=device
            )
            vec = model.encode_passage(ids)
            all_passage_vecs.append(vec.cpu().numpy().flatten())
    passage_matrix = np.array(all_passage_vecs)

    # ── Backup existing models before overwriting ──
    qa_files = [f for f in os.listdir(model_dir)
                if f.startswith('qa') and f.endswith(('.pt', '.pkl'))]
    if qa_files:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(model_dir, "backups", ts)
        os.makedirs(backup_dir, exist_ok=True)
        for f in qa_files:
            src = os.path.join(model_dir, f)
            shutil.copy2(src, os.path.join(backup_dir, f))
        print(f"  ⮡ Backed up {len(qa_files)} existing QA files → {backup_dir}")

    # ── Save everything ──
    model_path = os.path.join(model_dir, "qa_neural.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab.size,
        'total_params': total_params,
        'loss_history': loss_history,
    }, model_path)
    print(f"\n  ✓ Model saved: {model_path} ({os.path.getsize(model_path):,} bytes)")

    vocab_path = os.path.join(model_dir, "qa_vocab.pkl")
    vocab.save(vocab_path)
    print(f"  ✓ Vocab saved: {vocab_path}")

    index_path = os.path.join(model_dir, "qa_index.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump({
            'passage_matrix': passage_matrix,
            'corpus_chunks': corpus_chunks,
            'chunk_sources': chunk_sources,
            'max_seq_len': max_seq_len,
        }, f)
    print(f"  ✓ Index saved: {index_path} ({os.path.getsize(index_path):,} bytes)")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — Final Loss: {loss_history[-1]:.4f}")
    print(f"  Parameters: {total_params:,}")
    print(f"{'='*60}\n")

    return model


# ═══════════════════════════════════════════════════════════════
# 5. BUILT-IN TRANSPORT LAW KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════

TRANSPORT_FAQ = [
    {
        "keywords": ["fatal", "accident", "what is", "define", "definition", "fatal accident"],
        "answer": (
            "A fatal accident is defined as a road accident in which one or more persons "
            "die either at the spot of the accident or within 30 days of the accident as a "
            "result of injuries sustained in the accident. As per MoRTH guidelines, every "
            "fatal accident must be reported through a First Accident Report (FAR) within "
            "48 hours and investigated through a Detailed Accident Report (DAR) within 90 days."
        ),
        "source": "MoRTH Road Accident Guidelines / Motor Vehicles Act",
    },
    {
        "keywords": ["time limit", "far", "filing", "48 hours", "deadline", "when to file", "time limit for filing"],
        "answer": (
            "The First Accident Report (FAR) must be filed by the investigating police officer "
            "within 48 hours of a road accident. This is mandated under Section 159 of the Motor "
            "Vehicles Act, 1988 (as amended in 2019). The FAR must be submitted to the Motor "
            "Accidents Claims Tribunal (MACT). Failure to file within 48 hours may result in "
            "departmental action against the officer."
        ),
        "source": "Motor Vehicles Act, 1988 — Section 159",
    },
    {
        "keywords": ["responsible", "who", "filing", "report", "officer", "who files", "who is responsible"],
        "answer": (
            "The investigating police officer at the police station where the accident is reported "
            "is responsible for filing both the First Accident Report (FAR) and the Detailed "
            "Accident Report (DAR). Under Section 159 of the Motor Vehicles Act, the officer in "
            "charge of the police station must forward the FAR to the Claims Tribunal within 48 "
            "hours. The Station House Officer (SHO) or designated Investigating Officer (IO) is "
            "accountable for the accuracy and timeliness of these reports."
        ),
        "source": "Motor Vehicles Act, 1988 — Section 159",
    },
    {
        "keywords": ["information", "required", "accident report", "contents", "fields", "what information"],
        "answer": (
            "An accident report (FAR) must contain: (1) Date, time, and exact location of the accident; "
            "(2) FIR number and police station details; (3) Nature of accident — fatal, grievous injury, "
            "or property damage; (4) Number of persons killed and injured; (5) Vehicle types and "
            "registration numbers; (6) Type of collision (head-on, rear-end, hit-and-run, etc.); "
            "(7) Road type and weather conditions; (8) Probable cause of accident; (9) IPC/BNS sections "
            "charged; (10) Brief facts of the case. The DAR adds driver details, licence information, "
            "witness statements, site sketch, and full investigation findings."
        ),
        "source": "MoRTH Form-1 (FAR) / Form-VII (DAR)",
    },
    {
        "keywords": ["motor vehicles act", "mva", "what is", "about", "overview"],
        "answer": (
            "The Motor Vehicles Act, 1988 is the primary legislation governing road transport in India. "
            "It covers: (1) Licensing of drivers and conductors; (2) Registration of motor vehicles; "
            "(3) Control of transport vehicles (permits); (4) Traffic regulation and penalties; "
            "(5) Insurance of motor vehicles against third-party risk; (6) Claims Tribunals for "
            "accident compensation; (7) Offences and penalties for traffic violations. The Act was "
            "significantly amended in 2019 (Motor Vehicles Amendment Act) to increase penalties, "
            "improve road safety, and introduce electronic enforcement."
        ),
        "source": "Motor Vehicles Act, 1988 (as amended in 2019)",
    },
    {
        "keywords": ["hit and run", "hit-and-run", "rules", "compensation", "fleeing"],
        "answer": (
            "In hit-and-run cases: (1) The driver who flees is charged under Section 161 of the Motor "
            "Vehicles Act and Section 279/304A of the IPC (now BNS); (2) Compensation is provided from "
            "the Motor Vehicle Accident Fund — ₹2,00,000 for death and ₹50,000 for grievous injury "
            "(as per 2019 amendment); (3) The registered owner is liable unless they prove the vehicle "
            "was stolen; (4) Any witness or bystander should note the vehicle registration number and "
            "report to the nearest police station; (5) Hit-and-run is a non-bailable offence."
        ),
        "source": "Motor Vehicles Act, 1988 — Sections 161-164",
    },
    {
        "keywords": ["dar", "detailed", "90 days", "detailed accident report"],
        "answer": (
            "The Detailed Accident Report (DAR) must be filed within 90 days of the accident. "
            "It is more comprehensive than the FAR and includes: complete driver details, "
            "driving licence verification, vehicle fitness certificate status, insurance details, "
            "witness statements, site investigation sketch, photographs, cause analysis, "
            "and full prosecution details. The DAR is submitted to the Motor Accidents Claims "
            "Tribunal (MACT) as Form VII."
        ),
        "source": "MoRTH Guidelines — Form VII (DAR)",
    },
    {
        "keywords": ["ipc", "section", "279", "304a", "338", "penal", "bns", "charges"],
        "answer": (
            "Common IPC sections applied in road accidents: "
            "Section 279 — Rash driving on a public way (up to 6 months imprisonment); "
            "Section 304A — Causing death by negligence (up to 2 years imprisonment); "
            "Section 338 — Causing grievous hurt by act endangering life (up to 2 years); "
            "Section 337 — Causing hurt by act endangering life; "
            "Section 427 — Mischief causing damage. "
            "Under the new Bharatiya Nyaya Sanhita (BNS), these are now covered under "
            "Sections 281 (rash driving), 106 (death by negligence), and 125 (endangering life)."
        ),
        "source": "Indian Penal Code / Bharatiya Nyaya Sanhita",
    },
    {
        "keywords": ["compensation", "claim", "insurance", "mact", "tribunal", "amount"],
        "answer": (
            "Accident compensation is handled by the Motor Accidents Claims Tribunal (MACT). "
            "Key points: (1) Claims can be filed by the injured person, legal heirs of the deceased, "
            "or their authorized agent; (2) Third-party insurance is mandatory for all motor vehicles; "
            "(3) Compensation is calculated based on the victim's age, income, and multiplier factor "
            "as per the Supreme Court's Sarla Verma formula; (4) For hit-and-run cases, compensation "
            "comes from the Motor Vehicle Accident Fund; (5) There is no upper limit on compensation "
            "amount — it is decided by the tribunal based on evidence."
        ),
        "source": "Motor Vehicles Act, 1988 — Chapter X, XII",
    },
    {
        "keywords": ["road safety", "rules", "guidelines", "prevention", "measures"],
        "answer": (
            "Key road safety rules under Indian law: (1) Speed limits — 50 km/h in cities, "
            "varies on highways; (2) Seat belts mandatory for driver and front passenger; "
            "(3) Helmets mandatory for two-wheeler riders and pillion; (4) Drunk driving is a "
            "serious offence — BAC limit is 30mg per 100ml of blood; (5) Using mobile phone while "
            "driving is punishable; (6) Vehicles must carry valid insurance, fitness certificate, "
            "pollution-under-control certificate; (7) Overloading penalties have been increased "
            "significantly under the 2019 amendment."
        ),
        "source": "Motor Vehicles Act, 1988 / Central Motor Vehicles Rules",
    },
    {
        "keywords": ["penalty", "fine", "punishment", "drunk", "driving", "offence"],
        "answer": (
            "Key penalties under the Motor Vehicles (Amendment) Act, 2019: "
            "Drunk driving — ₹10,000 fine and/or 6 months imprisonment (first offence); "
            "Driving without licence — ₹5,000 fine; "
            "Driving without insurance — ₹2,000 fine and/or 3 months; "
            "Overspeeding — ₹1,000-₹2,000 for LMV, ₹2,000-₹4,000 for HMV; "
            "Not wearing seat belt — ₹1,000 fine; "
            "Not wearing helmet — ₹1,000 fine and 3-month licence suspension; "
            "Using mobile phone while driving — ₹5,000 fine; "
            "Rash driving — ₹5,000 fine and/or imprisonment."
        ),
        "source": "Motor Vehicles (Amendment) Act, 2019",
    },
    {
        "keywords": ["grievous", "injury", "hurt", "non-fatal", "serious"],
        "answer": (
            "A grievous injury accident is one where the victim suffers injuries defined as "
            "'grievous hurt' under Section 320 of the IPC — including fractures, loss of limb "
            "or sight, permanent disfigurement, or injuries endangering life. These accidents "
            "must be reported through the FAR and investigated under IPC Section 338 (causing "
            "grievous hurt by endangering life). The investigating officer must record the "
            "medical report from the treating hospital."
        ),
        "source": "IPC Section 320, 338 / Motor Vehicles Act",
    },
]


def _keyword_score(question: str, text: str) -> float:
    """BM25-style keyword overlap score between question and text."""
    q_words = set(re.findall(r'\b\w+\b', question.lower()))
    # Remove stop words
    stop = {'a','an','the','is','are','was','were','what','which','who','how',
            'when','where','why','do','does','did','in','on','at','to','for',
            'of','and','or','it','its','this','that','with','from','by','as','be'}
    q_words -= stop
    if not q_words:
        return 0.0
    t_words = set(re.findall(r'\b\w+\b', text.lower()))
    overlap = q_words & t_words
    return len(overlap) / len(q_words)


def _extract_best_sentences(passage: str, question: str, max_sentences: int = 4) -> str:
    """Extract the most relevant sentences from a passage for a given question."""
    sentences = re.split(r'(?<=[.!?;])\s+', passage.strip())
    if len(sentences) <= max_sentences:
        return passage.strip()

    # Score each sentence by keyword overlap with the question
    scored = []
    for i, sent in enumerate(sentences):
        kw_score = _keyword_score(question, sent)
        # Bonus for being near the beginning (context)
        position_bonus = 0.1 if i < 2 else 0.0
        scored.append((kw_score + position_bonus, i, sent))

    # Take best sentences, preserving original order
    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return " ".join(s[2] for s in top).strip()


def _deduplicate_results(results: list, threshold: float = 0.7) -> list:
    """Remove near-duplicate answers based on word overlap."""
    if not results:
        return results
    kept = [results[0]]
    for r in results[1:]:
        is_dup = False
        r_words = set(re.findall(r'\b\w+\b', r['answer'].lower()))
        for k in kept:
            k_words = set(re.findall(r'\b\w+\b', k['answer'].lower()))
            if not r_words or not k_words:
                continue
            overlap = len(r_words & k_words) / max(len(r_words | k_words), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


# ═══════════════════════════════════════════════════════════════
# 6. QA PREDICTOR (INFERENCE) — IMPROVED
# ═══════════════════════════════════════════════════════════════

class QAPredictor:
    """Load trained QA model and answer questions with improved quality."""

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.passage_matrix = None
        self.corpus_chunks = []
        self.chunk_sources = []
        self.max_seq_len = 64

    def load(self):
        """Load model, vocab, and pre-computed index."""
        # Vocab
        self.vocab = QAVocabulary()
        self.vocab.load(os.path.join(self.model_dir, "qa_vocab.pkl"))

        # Model
        checkpoint = torch.load(
            os.path.join(self.model_dir, "qa_neural.pt"),
            map_location=self.device, weights_only=False
        )
        self.model = NeuralQAModel(
            vocab_size=checkpoint['vocab_size'],
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Index
        with open(os.path.join(self.model_dir, "qa_index.pkl"), 'rb') as f:
            index = pickle.load(f)
        self.passage_matrix = index['passage_matrix']
        self.corpus_chunks = index['corpus_chunks']
        self.chunk_sources = index['chunk_sources']
        self.max_seq_len = index.get('max_seq_len', 64)

        print(f"QA model loaded ({checkpoint['total_params']:,} params, "
              f"{len(self.corpus_chunks)} passages)")

    def _check_knowledge_base(self, question: str) -> list:
        """Check built-in knowledge base for high-confidence answers."""
        q_lower = question.lower().strip()
        scored_faqs = []

        for faq in TRANSPORT_FAQ:
            score = 0.0
            # Check keyword matches
            for kw in faq["keywords"]:
                if kw in q_lower:
                    # Multi-word keyword matches are worth more
                    word_count = len(kw.split())
                    score += 0.3 * word_count

            # Also check general word overlap
            kw_score = _keyword_score(question, faq["answer"])
            score += kw_score * 0.3

            if score > 0.25:
                scored_faqs.append((score, faq))

        scored_faqs.sort(key=lambda x: x[0], reverse=True)
        return scored_faqs

    def answer(self, question: str, top_k: int = 3) -> list:
        """
        Find most relevant answers for a question.

        Strategy:
        1. Check built-in knowledge base for high-confidence FAQ matches
        2. Run neural retrieval on the corpus
        3. Re-rank neural results with keyword boosting
        4. Extract best sentences from passages
        5. Merge FAQ + neural results, deduplicate
        """
        results = []

        # ── Step 1: Knowledge base lookup ──
        kb_matches = self._check_knowledge_base(question)
        for score, faq in kb_matches[:2]:  # Max 2 from KB
            results.append({
                "answer": faq["answer"],
                "score": round(min(score + 0.5, 1.0), 4),  # Boost KB scores
                "source": faq["source"],
            })

        # ── Step 2: Neural retrieval (if model is loaded) ──
        if self.model is not None:
            query_ids = torch.tensor(
                [self.vocab.encode(question, self.max_seq_len)],
                dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                query_vec = self.model.encode_query(query_ids).cpu().numpy().flatten()

            # Cosine similarity against all passages
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            passage_norms = self.passage_matrix / (
                np.linalg.norm(self.passage_matrix, axis=1, keepdims=True) + 1e-8
            )
            neural_scores = passage_norms @ query_norm

            # Get top candidates (more than we need for re-ranking)
            candidate_count = min(top_k * 4, len(neural_scores))
            top_indices = neural_scores.argsort()[-candidate_count:][::-1]

            # ── Step 3: Re-rank with keyword overlap ──
            reranked = []
            for idx in top_indices:
                n_score = float(neural_scores[idx])
                kw_score = _keyword_score(question, self.corpus_chunks[idx])
                # Combined score: 60% neural + 40% keyword
                combined = 0.6 * n_score + 0.4 * kw_score
                reranked.append((combined, idx))

            reranked.sort(key=lambda x: x[0], reverse=True)

            # ── Step 4: Extract best sentences from top passages ──
            for combined_score, idx in reranked[:top_k]:
                raw_passage = self.corpus_chunks[idx]
                # Extract the most relevant sentences from the chunk
                extracted = _extract_best_sentences(raw_passage, question)
                if len(extracted) > 30:  # Skip very short fragments
                    results.append({
                        "answer": extracted,
                        "score": round(combined_score, 4),
                        "source": self.chunk_sources[idx],
                    })

        # ── Step 5: Deduplicate and rank ──
        results.sort(key=lambda x: x["score"], reverse=True)
        results = _deduplicate_results(results, threshold=0.6)

        # Return top_k results
        final = results[:top_k]
        return final if final else [{"answer": "No relevant information found for this question.",
                                     "score": 0, "source": ""}]


# ═══════════════════════════════════════════════════════════════
# 7. REGEX-BASED TEXT EXTRACTION (FALLBACK — NO MODEL NEEDED)
# ═══════════════════════════════════════════════════════════════

def extract_from_text_regex(text: str) -> dict:
    """
    Extract accident report fields from natural language text using
    regex and keyword matching. Works without any trained model.
    Returns a dict with all FAR/DAR fields populated where possible.
    """
    data = {}
    text_lower = text.lower()

    # FIR Number
    m = re.search(r'(?:FIR|F\.?I\.?R\.?)\s*(?:No\.?|number|num)?\s*[:\-]?\s*([A-Za-z0-9/\-]+/\d{4})', text, re.I)
    if m:
        data['fir_no'] = m.group(1).strip()
    else:
        m = re.search(r'(?:FIR|F\.?I\.?R\.?)\s*(?:No\.?|number|num)?\s*[:\-]?\s*([A-Za-z0-9/\-]+)', text, re.I)
        if m:
            data['fir_no'] = m.group(1).strip()

    # Date of accident
    date_pattern = r'(?:occurred|happened|took place|accident)\s+(?:on\s+)?(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[,]?\s*)?(?:on\s+)?(\d{1,2}[\s/\-](?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s/\-]\d{2,4})'
    m = re.search(date_pattern, text, re.I)
    if m:
        data['date_of_accident'] = m.group(1).strip()
    else:
        m = re.search(r'(?:on|dated?)\s+(\d{1,2}[\s/\-](?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{4})', text, re.I)
        if m:
            data['date_of_accident'] = m.group(1).strip()

    # Date of registration/report
    m = re.search(r'(?:registered|filed|reported)\s+(?:on\s+)?(\d{1,2}[\s/\-](?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+\d{4})', text, re.I)
    if m:
        data['report_date'] = m.group(1).strip()

    # Time of accident
    m = re.search(r'(?:at|around|approximately)\s+(\d{1,2}[:\.]\d{2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?)', text, re.I)
    if m:
        data['time_of_accident'] = m.group(1).strip()

    # Time of reporting
    m = re.search(r'(?:reported|informed|intimated)\s+(?:at|around)\s+(\d{1,2}[:\.]\d{2}\s*(?:AM|PM|am|pm)?)', text, re.I)
    if m:
        data['time_of_reporting'] = m.group(1).strip()

    # Police Station
    m = re.search(r'(?:at|from)\s+([A-Z][a-zA-Z\s]+?)\s*(?:Police Station|PS|Thana)', text)
    if m:
        data['police_station'] = m.group(1).strip() + ' Police Station'

    # District / City
    m = re.search(r'(?:in|of|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:district|city)?\s*[,]?\s*(?:in\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    # More targeted: look for "in <City>, <State>" or "<Place>, <District>"
    m2 = re.search(r'in\s+([A-Z][a-zA-Z\s]+?),\s*([A-Z][a-zA-Z\s]+?)(?:\.|,|$)', text)
    if m2:
        data.setdefault('district', m2.group(1).strip())

    # State
    states = ['Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh',
              'Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka',
              'Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram',
              'Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu',
              'Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
              'Delhi','Jammu and Kashmir']
    for st in states:
        if st.lower() in text_lower:
            data['state'] = st
            break

    # Place of accident (road/location)
    m = re.search(r'(?:on|near|at)\s+((?:State Highway|National Highway|NH|SH)\s*\d+[A-Za-z]?\s+(?:near\s+)?[A-Za-z\s]+?)(?:in|,|\.|$)', text, re.I)
    if m:
        data['place_of_accident'] = m.group(1).strip()
    else:
        m = re.search(r'near\s+([A-Z][a-zA-Z\s]+?)\s+(?:in|on|,)', text)
        if m:
            data['place_of_accident'] = m.group(1).strip()

    # IPC / BNS Sections
    ipc_matches = re.findall(r'(?:Section|Sec\.?)s?\s+(\d{2,3}[A-Za-z]?)(?:\s*(?:and|,|&)\s*(\d{2,3}[A-Za-z]?))*', text, re.I)
    sections = []
    for match_tuple in ipc_matches:
        for s in match_tuple:
            if s:
                sections.append(s)
    # Also find individual section numbers near "IPC" or "BNS"
    ipc_nums = re.findall(r'(?:Section|Sec)s?\s+([\d]+[A-Za-z]?(?:\s*(?:and|,|&)\s*[\d]+[A-Za-z]?)*)', text, re.I)
    for block in ipc_nums:
        for s in re.findall(r'(\d+[A-Za-z]?)', block):
            if s not in sections:
                sections.append(s)
    if sections:
        data['ipc_sections'] = ', '.join(sections)

    # Nature of accident
    if any(w in text_lower for w in ['fatal', 'died', 'killed', 'death', 'dead']):
        data['nature_of_accident'] = 'Fatal'
    elif any(w in text_lower for w in ['grievous', 'serious', 'critical']):
        data['nature_of_accident'] = 'Grievous Injury'
    elif any(w in text_lower for w in ['injured', 'hurt', 'wound']):
        data['nature_of_accident'] = 'Injury'
    else:
        data['nature_of_accident'] = 'Damage/Loss of property'

    # Number killed
    word_nums = {'one':'1','two':'2','three':'3','four':'4','five':'5',
                 'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
                 'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14',
                 'fifteen':'15','sixteen':'16','seventeen':'17','eighteen':'18',
                 'nineteen':'19','twenty':'20','thirty':'30','forty':'40','fifty':'50'}
    # Compound numbers: thirty-six, twenty-three, etc.
    compound_nums = {}
    tens = {'twenty':'2','thirty':'3','forty':'4','fifty':'5','sixty':'6','seventy':'7','eighty':'8','ninety':'9'}
    ones = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9'}
    for t_word, t_val in tens.items():
        for o_word, o_val in ones.items():
            compound_nums[f'{t_word}-{o_word}'] = f'{t_val}{o_val}'
            compound_nums[f'{t_word} {o_word}'] = f'{t_val}{o_val}'
    all_nums = {**word_nums, **compound_nums}
    m = re.search(r'(\d+|' + '|'.join(sorted(all_nums.keys(), key=len, reverse=True)) + r')\s+(?:persons?\s+)?(?:died|killed|dead|fatalities|death)', text_lower)
    if m:
        v = m.group(1)
        data['num_fatalities'] = all_nums.get(v, v)
    else:
        m = re.search(r'(?:total of|about|approximately)?\s*(\d+|' + '|'.join(sorted(all_nums.keys(), key=len, reverse=True)) + r')\s+(?:persons?\s+)?(?:died|killed|dead)', text_lower)
        if m:
            v = m.group(1)
            data['num_fatalities'] = all_nums.get(v, v)

    # Number injured — try compound numbers first
    compound_pattern = '|'.join(sorted(all_nums.keys(), key=len, reverse=True))
    m = re.search(r'(?:about|approximately|around)?\s*(' + compound_pattern + r'|\d+)\s*(?:were\s+|persons?\s+)?(?:injured|hurt|wounded|hospitalised|hospitalized)', text_lower)
    if m:
        v = m.group(1).strip()
        data['num_injured'] = all_nums.get(v, v)

    # Number of vehicles
    m = re.search(r'(\d+|two|three|four|five)\s+vehicles?', text_lower)
    if m:
        v = m.group(1)
        data['num_vehicles'] = word_nums.get(v, v)

    # Vehicle registrations
    reg_matches = re.findall(r'([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{4})', text)
    if reg_matches:
        data['v1_reg'] = reg_matches[0] if len(reg_matches) > 0 else ''
        data['v2_reg'] = reg_matches[1] if len(reg_matches) > 1 else ''

    # Vehicle types  
    vehicle_types = []
    vtype_patterns = [
        r'(?:first|1st)\s+vehicle\s+was\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\s+bearing|,|\.|$)',
        r'(?:second|2nd)\s+vehicle\s+was\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\s+bearing|,|\.|$)',
    ]
    for vp in vtype_patterns:
        m = re.search(vp, text, re.I)
        if m:
            vehicle_types.append(m.group(1).strip())
    # Fallback: look for known vehicle types
    known_vehicles = ['passenger bus', 'bus', 'truck', 'car', 'auto rickshaw', 
                      'motorcycle', 'bike', 'scooter', 'lorry', 'tanker', 'trailer',
                      'tractor', 'van', 'jeep', 'taxi', 'SUV', 'tempo', 'ambulance']
    if not vehicle_types:
        for vt in known_vehicles:
            if vt.lower() in text_lower:
                if vt not in vehicle_types:
                    vehicle_types.append(vt)
    data['v1_type'] = vehicle_types[0] if len(vehicle_types) > 0 else ''
    data['v2_type'] = vehicle_types[1] if len(vehicle_types) > 1 else ''

    # Driver names
    driver_matches = re.findall(r'driven\s+by\s+([A-Z][a-zA-Z\s]+?)(?:\s+from|,|\.|\s+aged|\s+residing)', text)
    if driver_matches:
        data['v1_driver'] = driver_matches[0].strip() if len(driver_matches) > 0 else ''
        data['v2_driver'] = driver_matches[1].strip() if len(driver_matches) > 1 else ''

    # Owner info
    owner_matches = re.findall(r'owned\s+by\s+([A-Z][a-zA-Z\s]+?)(?:\s+and|,|\.|$)', text)
    if owner_matches:
        data['v1_owner'] = owner_matches[0].strip() if len(owner_matches) > 0 else ''
        data['v2_owner'] = owner_matches[1].strip() if len(owner_matches) > 1 else ''

    # Insurance
    ins_matches = re.findall(r'insured\s+with\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)', text)
    if ins_matches:
        data['v1_insurance'] = ins_matches[0].strip() if len(ins_matches) > 0 else ''
        data['v2_insurance'] = ins_matches[1].strip() if len(ins_matches) > 1 else ''

    # Collision type
    collision_types = ['head-on collision', 'head on collision', 'rear-end', 'rear end',
                       'side collision', 'T-bone', 'hit and run', 'hit-and-run',
                       'vehicle-to-vehicle', 'vehicle to vehicle', 'pile-up', 'pile up',
                       'overturn', 'overturned', 'rollover', 'skidding']
    for ct in collision_types:
        if ct.lower() in text_lower:
            data['collision_type'] = ct.title()
            break

    # Nature of collision
    m = re.search(r'(?:nature of collision|collision was)\s+(?:was\s+)?(?:a\s+)?([a-zA-Z\-\s]+?)(?:\.|,|$)', text, re.I)
    if m:
        data['collision_nature'] = m.group(1).strip()

    # Cause of accident
    causes = []
    cause_keywords = ['overspeeding', 'rash driving',
                      'drunk driving', 'drunken driving', 'negligence', 'wrong side',
                      'dangerous overtaking', 'brake failure', 'tyre burst',
                      'signal jumping', 'mobile phone', 'overloading', 'poor visibility']
    for ck in cause_keywords:
        if ck in text_lower:
            causes.append(ck.title())
    if causes:
        data['cause'] = ', '.join(causes)

    # Weather
    weather_words = {'clear': 'Clear', 'rain': 'Rain', 'heavy rain': 'Heavy Rain',
                     'fog': 'Fog', 'mist': 'Mist', 'storm': 'Storm',
                     'cloudy': 'Cloudy', 'hail': 'Hail'}
    for wk, wv in weather_words.items():
        if wk in text_lower:
            data['weather'] = wv
            break

    # Lighting
    light_words = {'daylight': 'Daylight', 'night': 'Night', 'dusk': 'Dusk',
                   'dawn': 'Dawn', 'dark': 'Dark'}
    for lk, lv in light_words.items():
        if lk in text_lower:
            data['lighting'] = lv
            break

    # Visibility
    m = re.search(r'visibility\s+(?:was\s+)?(?:above|over|more than|about)?\s*(\d+\s*(?:meters?|metres?|m))', text, re.I)
    if m:
        data['visibility'] = m.group(1).strip()

    # Road type
    road_types = {'state highway': 'State Highway', 'national highway': 'National Highway',
                  'expressway': 'Expressway', 'district road': 'District Road',
                  'village road': 'Village Road', 'city road': 'City Road'}
    for rk, rv in road_types.items():
        if rk in text_lower:
            data['road_type'] = rv
            break

    # Jurisdiction
    juris_words = {'panchayat': 'Panchayat', 'municipal': 'Municipal Corporation',
                   'nagar palika': 'Nagar Palika', 'cantonment': 'Cantonment'}
    for jk, jv in juris_words.items():
        if jk in text_lower:
            data['jurisdiction'] = jv
            break

    # Area type
    area_words = {'open area': 'Open Area', 'residential': 'Residential',
                  'commercial': 'Commercial', 'industrial': 'Industrial'}
    for ak, av in area_words.items():
        if ak in text_lower:
            data['area_type'] = av
            break

    # Load condition
    load_words = {'normally loaded': 'Normally Loaded', 'overloaded': 'Overloaded',
                  'empty': 'Empty', 'not loaded': 'Not Loaded'}
    for lk, lv in load_words.items():
        if lk in text_lower:
            data['load_condition'] = lv
            break

    # Investigating officer
    m = re.search(r'(?:investigating officer|IO|I\.O\.)\s+(?:was\s+)?(?:SI|Sub Inspector|Inspector|ASI|SHO)?\s*\.?\s*([A-Z][a-zA-Z\s]+?)(?:,|\.|\s+PIS|\s+attached|$)', text)
    if m:
        data['officer_name'] = m.group(1).strip()

    # PIS number
    m = re.search(r'PIS\s+(?:number|no\.?)?\s*[:\-]?\s*(\d+)', text, re.I)
    if m:
        data['officer_pis'] = m.group(1).strip()

    # Informant
    m = re.search(r'informant\s+was\s+(?:Constable|SI|ASI|SHO|Inspector)?\s*([A-Z][a-zA-Z\s]+?)(?:,|\.|\s+residing)', text)
    if m:
        data['informant_name'] = m.group(1).strip()

    # Source of information
    source_words = {'police': 'Police', 'hospital': 'Hospital', 'witness': 'Witness',
                    'public': 'Public', 'media': 'Media'}
    for sk, sv in source_words.items():
        if f'source of information was {sk}' in text_lower or f'source was {sk}' in text_lower:
            data['info_source'] = sv
            break

    # Hospital
    m = re.search(r'(?:taken to|admitted to|treated at|rushed to)\s+([A-Z][a-zA-Z\s]+?(?:Hospital|Medical|Centre|Center|Clinic))', text)
    if m:
        data['hospital'] = m.group(1).strip()

    # Doctor
    m = re.search(r'(?:Dr\.?|Doctor)\s+([A-Z][a-zA-Z\.\s]+?)(?:\s+attended|,|\.|$)', text)
    if m:
        data['doctor'] = 'Dr. ' + m.group(1).strip()

    # CCTV
    if 'no cctv' in text_lower or 'cctv footage was available' not in text_lower:
        if 'no cctv' in text_lower or 'not available' in text_lower:
            data['cctv'] = 'No'
    if 'cctv footage' in text_lower and 'no' not in text_lower.split('cctv')[0][-20:]:
        data['cctv'] = 'Yes'
    if 'no cctv' in text_lower:
        data['cctv'] = 'No'

    # Vehicles impounded
    if 'impounded' in text_lower:
        data['vehicles_impounded'] = 'Yes'

    # Drivers found
    if 'drivers were found' in text_lower or 'driver was found' in text_lower:
        data['drivers_found'] = 'Yes'
    elif 'driver fled' in text_lower or 'absconding' in text_lower:
        data['drivers_found'] = 'No'

    return data


# ═══════════════════════════════════════════════════════════════
# 8. FORM GENERATOR
# ═══════════════════════════════════════════════════════════════

class FormGenerator:
    """Generate editable FAR/DAR form drafts with regex and NER support."""

    FAR_TEMPLATE = {
        "form_title": "FORM 1 — FIRST ACCIDENT REPORT (FAR)",
        "subtitle": "By Investigating Officer to Claims Tribunal within 48 hours",
        "fields": [
            {"id": "fir_no", "label": "FIR No.", "value": "", "type": "text"},
            {"id": "date_of_accident", "label": "1. Date of Accident", "value": "", "type": "date"},
            {"id": "time_of_accident", "label": "2. Time of Accident", "value": "", "type": "time"},
            {"id": "place_of_accident", "label": "3. Place of Accident", "value": "", "type": "text"},
            {"id": "nature_of_accident", "label": "5. Nature of Accident", "value": "",
             "type": "select", "options": ["Injury", "Fatal", "Damage/Loss of property", "Other"]},
            {"id": "num_vehicles", "label": "   Vehicles Involved", "value": "", "type": "number"},
            {"id": "num_fatalities", "label": "   Fatalities", "value": "0", "type": "number"},
            {"id": "num_injured", "label": "   Injured", "value": "0", "type": "number"},
            {"id": "v1_type", "label": "8. Vehicle 1 Type", "value": "", "type": "text"},
            {"id": "v2_type", "label": "   Vehicle 2 Type", "value": "", "type": "text"},
            {"id": "collision_type", "label": "10. Collision Type", "value": "", "type": "text"},
            {"id": "weather", "label": "   Weather Condition", "value": "", "type": "text"},
            {"id": "road_type", "label": "   Road Classification", "value": "", "type": "text"},
            {"id": "cause", "label": "   Cause of Accident", "value": "", "type": "text"},
            {"id": "officer_name", "label": "Investigating Officer", "value": "", "type": "text"},
            {"id": "report_date", "label": "Date of Report", "value": "", "type": "date"},
        ]
    }

    DAR_TEMPLATE = {
        "form_title": "FORM VII — DETAILED ACCIDENT REPORT (DAR)",
        "subtitle": "By Investigating Officer to Claims Tribunal within 90 days",
        "fields": [
            {"id": "fir_no", "label": "FIR No.", "value": "", "type": "text"},
            {"id": "date_of_accident", "label": "1. Date of Accident", "value": "", "type": "date"},
            {"id": "time_of_accident", "label": "2. Time of Accident", "value": "", "type": "time"},
            {"id": "place_of_accident", "label": "3. Place of Accident", "value": "", "type": "text"},
            {"id": "nature_fatal", "label": "4. Fatal", "value": "", "type": "select", "options": ["Yes", "No"]},
            {"id": "nature_grievous", "label": "   Grievous Injury", "value": "", "type": "select", "options": ["Yes", "No"]},
            {"id": "v_reg_no", "label": "5. Offending Vehicle Reg. No.", "value": "", "type": "text"},
            {"id": "v_type", "label": "   Vehicle Type", "value": "", "type": "text"},
            {"id": "driver_name", "label": "6. Driver Name", "value": "", "type": "text"},
            {"id": "driver_license", "label": "   Driving Licence No.", "value": "", "type": "text"},
            {"id": "driver_alcohol", "label": "13. Under influence of alcohol", "value": "", "type": "select", "options": ["Yes", "No"]},
            {"id": "victim_name", "label": "21. Victim Name", "value": "", "type": "text"},
            {"id": "victim_age", "label": "22. Victim Age", "value": "", "type": "number"},
            {"id": "ipc_sections", "label": "30. IPC Sections Charged", "value": "", "type": "text"},
            {"id": "description", "label": "31. Detailed Description", "value": "", "type": "textarea"},
            {"id": "officer_name", "label": "S.H.O/I.O Name", "value": "", "type": "text"},
            {"id": "report_date", "label": "Date", "value": "", "type": "date"},
        ]
    }

    def get_far_template(self):
        import copy; return copy.deepcopy(self.FAR_TEMPLATE)

    def get_dar_template(self):
        import copy; return copy.deepcopy(self.DAR_TEMPLATE)

    def prefill_far(self, entities: dict):
        import copy
        form = copy.deepcopy(self.FAR_TEMPLATE)
        mapping = {
            "date_of_accident": "date_of_accident",
            "time_of_accident": "time_of_accident",
            "place_of_accident": "place_of_accident",
            "num_fatalities": "number_of_fatalities",
            "num_injured": "number_of_injured",
            "v1_type": "vehicle_1",
            "v2_type": "vehicle_2",
            "collision_type": "collision_type",
            "weather": "weather_condition",
            "road_type": "road_type",
            "cause": "cause_of_accident",
        }
        for field in form["fields"]:
            ekey = mapping.get(field["id"])
            if ekey and ekey in entities and entities[ekey]:
                field["value"] = str(entities[ekey])
        sev = entities.get("nature_of_accident", "")
        if sev:
            for f in form["fields"]:
                if f["id"] == "nature_of_accident":
                    f["value"] = sev
        return form

    def prefill_dar(self, entities: dict):
        """Pre-fill DAR template from extracted entities dict."""
        import copy
        form = copy.deepcopy(self.DAR_TEMPLATE)
        mapping = {
            "fir_no": "fir_no",
            "date_of_accident": "date_of_accident",
            "time_of_accident": "time_of_accident",
            "place_of_accident": "place_of_accident",
            "v_reg_no": "v1_reg",
            "v_type": "v1_type",
            "driver_name": "v1_driver",
            "ipc_sections": "ipc_sections",
            "officer_name": "officer_name",
            "report_date": "report_date",
        }
        for field in form["fields"]:
            ekey = mapping.get(field["id"])
            if ekey and ekey in entities and entities[ekey]:
                field["value"] = str(entities[ekey])
        # Nature fatal
        nature = entities.get("nature_of_accident", "")
        if nature:
            for f in form["fields"]:
                if f["id"] == "nature_fatal":
                    f["value"] = "Yes" if nature == "Fatal" else "No"
                if f["id"] == "nature_grievous":
                    f["value"] = "Yes" if nature == "Grievous Injury" else "No"
        return form

    def prefill_far_from_regex(self, text: str):
        """Extract fields using regex and prefill FAR template."""
        data = extract_from_text_regex(text)
        import copy
        form = copy.deepcopy(self.FAR_TEMPLATE)
        mapping = {
            "fir_no": "fir_no",
            "date_of_accident": "date_of_accident",
            "time_of_accident": "time_of_accident",
            "place_of_accident": "place_of_accident",
            "nature_of_accident": "nature_of_accident",
            "num_vehicles": "num_vehicles",
            "num_fatalities": "num_fatalities",
            "num_injured": "num_injured",
            "v1_type": "v1_type",
            "v2_type": "v2_type",
            "collision_type": "collision_type",
            "weather": "weather",
            "road_type": "road_type",
            "cause": "cause",
            "officer_name": "officer_name",
            "report_date": "report_date",
        }
        for field in form["fields"]:
            ekey = mapping.get(field["id"])
            if ekey and ekey in data and data[ekey]:
                field["value"] = str(data[ekey])
        return form, data

    def prefill_dar_from_regex(self, text: str):
        """Extract fields using regex and prefill DAR template."""
        data = extract_from_text_regex(text)
        form = self.prefill_dar(data)
        return form, data

    def form_to_text(self, form):
        lines = [form["form_title"], form["subtitle"], "=" * 60]
        for f in form["fields"]:
            val = f.get("value", "") or "___________"
            lines.append(f"{f['label']}: {val}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 8. MAIN — TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("PyTorch QA model defined successfully.")
    # Quick parameter count check
    vocab_size = 5000
    model = NeuralQAModel(vocab_size=vocab_size)
    print(f"Parameters: {model.count_parameters():,}")
