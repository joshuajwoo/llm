import os
import re

# We will use Python's built-in re module.
# The standard GPT-4 regex pattern uses \p{L} which is not supported by 're'
# We will use an equivalent robust pattern that works with standard 're'.
# It splits on contractions, words, non-whitespace punctuation, and space sequences.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""

def get_stats(ids, counts=None):
    """Count the frequency of consecutive pairs of integers."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """Replace all consecutive occurrences of `pair` in `ids` with `idx`."""
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256 - 2 # 256 bytes + 2 special tokens
        self.merges = {} # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)} # int -> bytes
        
        # Special tokens
        self.special_tokens = {
            "<|endoftext|>": vocab_size - 2,
            "<|pad|>": vocab_size - 1
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")
            
        self.pattern = re.compile(GPT2_SPLIT_PATTERN, re.IGNORECASE)

    def train(self, text, verbose=False):
        """Train the BPE tokenizer on the given text."""
        # Split text into chunks using regex
        text_chunks = re.findall(self.pattern, text)
        
        # Convert chunks to raw bytes, then to lists of integers
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        # Iteratively merge the most common pair
        for i in range(self.num_merges):
            # Count pair frequencies across all chunks
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            
            if not stats:
                break # No more pairs to merge
                
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            
            # Record the merge
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply the merge to all chunks
            ids = [merge(chunk_ids, best_pair, new_idx) for chunk_ids in ids]
            
            if verbose and i % 500 == 0:
                print(f"Merge {i+1}/{self.num_merges}: {best_pair} -> {new_idx}")

    def encode(self, text, allowed_special=set()):
        """Encode a string into a list of token IDs."""
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                # Find the pair in chunk_ids that was merged earliest during training
                stats = get_stats(chunk_ids)
                # Filter to pairs we actually know about
                valid_pairs = {pair: self.merges[pair] for pair in stats if pair in self.merges}
                if not valid_pairs:
                    break # No more valid merges
                
                # Get the pair with the lowest merge index (happened earliest)
                best_pair = min(valid_pairs, key=valid_pairs.get)
                new_idx = valid_pairs[best_pair]
                
                chunk_ids = merge(chunk_ids, best_pair, new_idx)
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        """Decode a list of token IDs back into a string."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # errors="replace" ensures that partial utf-8 byte sequences don't crash the decoder
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_prefix):
        """Save the merges to disk."""
        os.makedirs(os.path.dirname(file_prefix) if os.path.dirname(file_prefix) else ".", exist_ok=True)
        with open(file_prefix + ".merges", "w", encoding="utf-8") as f:
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")
                
    def load(self, file_prefix):
        """Load the merges from disk and reconstruct the vocab."""
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        with open(file_prefix + ".merges", "r", encoding="utf-8") as f:
            for line in f:
                p0, p1, idx = map(int, line.split())
                self.merges[(p0, p1)] = idx
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
                
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")


if __name__ == "__main__":
    from config import ModelConfig
    cfg = ModelConfig()

    print("Loading data...")
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Text length: {len(text)} chars")
    print(f"Training tokenizer to vocab size {cfg.vocab_size}...")
    
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tokenizer.train(text, verbose=True)
    
    # Save it
    tokenizer.save("data/tokenizer")
    print("Tokenizer saved to data/tokenizer.merges")
    
    # Test round trip
    test_str = "The quick brown fox jumps over the lazy dog. Let's test math: $\\sum_{i=1}^n x_i$ ∇"
    ids = tokenizer.encode(test_str)
    decoded = tokenizer.decode(ids)
    
    print(f"\nOriginal: {test_str}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded:  {decoded}")
    assert test_str == decoded, "Round trip failed!"
    print("\n✅ Round trip test passed!")
