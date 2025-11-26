import random

from torch.utils.data import Dataset


class SentenceCorpus:
    """
    Holder for a corpus of sentences with their document and position info.
    Input: list of documents, where each document is list[str] (sentences in order).
    This lets us define 'neighbor' sentences (within a distance window).
    """

    docs: list[list[str]]
    flat: list[tuple[int, int, str]]  # list of (doc_id, sent_idx, text)
    N: int

    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        # flatten to a list of (doc_id, sent_idx, text)
        self.flat = []
        for d_id, doc in enumerate(docs):
            for s_idx, text in enumerate(doc):
                self.flat.append((d_id, s_idx, text))
        self.N = len(self.flat)

    def get_sentence(self, flat_idx: int) -> str:
        return self.flat[flat_idx][2]

    def neighbor_indices(self, flat_idx: int, max_dist: int = 10) -> list[int]:
        """Return indices of sentences within distance < max_dist in same document (excluding itself)."""
        d_id, s_idx, _ = self.flat[flat_idx]
        doc = self.docs[d_id]
        lo = max(0, s_idx - max_dist + 1)
        hi = min(len(doc) - 1, s_idx + max_dist - 1)
        if lo <= hi:
            for j in range(lo, hi + 1):
                if j == s_idx:
                    continue
                # map (d_id, j) back to flat index: we can scan (cheap once) or build index map
                # Let's build mapping once:
            # we'll use a mapping built in constructor
        return []  # unused; dataset uses precomputed mapping


class ContrastiveSentenceDataset(Dataset):
    """
    Dataset returning an anchor index. The collate function constructs multiple text inputs:
      - anchor_text (used twice for dropout augmentations in model forward)
      - neighbor_texts (0..k positives from nearby sentences)
    The collator will produce tokenized batches.
    """

    docs: list[list[str]]
    corpus: SentenceCorpus
    neighbor_window: int
    num_neighbors: int
    sample_neighbors_prob: float
    N: int

    def __init__(
        self,
        docs: list[list[str]],
        neighbor_window: int = 10,
        num_neighbors: int = 1,
        sample_neighbors_prob: float = 1.0,
    ):
        """
        docs: list of documents (each is list of sentences)
        neighbor_window: maximum sentence distance to consider neighbor (distance < neighbor_window)
        num_neighbors: number of neighbor positives to sample per anchor (if available)
        sample_neighbors_prob: probability to include neighbor positives (for mixing supervised/unsupervised)
        """
        self.docs = docs
        self.corpus = SentenceCorpus(docs)
        self.num_neighbors = num_neighbors
        self.neighbor_window = neighbor_window
        self.sample_neighbors_prob = sample_neighbors_prob

        # build mapping (doc_id, sent_idx) -> flat index for quick neighbor lookup
        self.doc_index_starts = []
        flat_idx = 0
        for doc in docs:
            self.doc_index_starts.append(flat_idx)
            flat_idx += len(doc)

        # map (doc_id, sent_idx) -> flat_index:
        self.doc_pos_to_flat = {}
        flat = 0
        for d_id, doc in enumerate(docs):
            for s_idx, _ in enumerate(doc):
                self.doc_pos_to_flat[(d_id, s_idx)] = flat
                flat += 1
        self.N = flat

    def __len__(self):
        return self.N

    def sample_neighbors_for_flat_idx(self, flat_idx: int) -> list[int]:
        d_id, s_idx, _ = self.corpus.flat[flat_idx]
        doc = self.docs[d_id]
        lo = max(0, s_idx - (self.neighbor_window - 1))
        hi = min(len(doc) - 1, s_idx + (self.neighbor_window - 1))
        candidates = [j for j in range(lo, hi + 1) if j != s_idx]
        if not candidates:
            return []
        k = min(self.num_neighbors, len(candidates))
        sampled = random.sample(candidates, k)
        flat_sampled = [self.doc_pos_to_flat[(d_id, j)] for j in sampled]
        return flat_sampled

    def __getitem__(self, idx: int):
        """
        Returns an item describing the anchor and indices of positives:
          {
            'anchor_idx': int,
            'anchor_text': str,
            'neighbor_pos_indices': List[int]  # could be empty
          }
        """
        anchor_text = self.corpus.get_sentence(idx)
        neighbor_indices = []
        if random.random() < self.sample_neighbors_prob:
            neighbor_indices = self.sample_neighbors_for_flat_idx(idx)
        return {"anchor_idx": idx, "anchor_text": anchor_text, "neighbor_pos_indices": neighbor_indices}
