import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
import random
import re
import requests
import os

# Step 1: Load and Preprocess Corpus
def load_corpus(file_path=None, url=None):
    """
    Load text from a local file or URL. If neither provided, use a sample play-like text.
    """
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return []
    elif file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
    else:
        # Sample play-like text
        text = """
        ROMEO: But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun.
        JULIET: O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name.
        ROMEO: Shall I hear more, or shall I speak at this? Call me but love, and I'll be new baptized.
        JULIET: 'Tis but thy name that is my enemy. Thou art thyself, though not a Montague.
        NARRATOR: The scene unfolds under the moonlit Verona sky, where lovers whisper their vows.
        """ * 100

    # Preprocess: Remove Project Gutenberg headers/footers
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if start_marker in text:
        text = text.split(start_marker, 1)[1]
    if end_marker in text:
        text = text.split(end_marker, 1)[0]

    # Normalize whitespace, convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # Split into sentences using multiple delimiters
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip() and len(s.split()) > 1]
    print(f"Loaded {len(sentences)} sentences from corpus.")
    print(f"Sample sentences: {sentences[:2]}")
    return sentences

# Step 2: BPE Tokenizer
class BPETokenizer:
    def __init__(self, corpus, vocab_size=1000):
        self.corpus = corpus
        self.vocab_size = max(vocab_size, 100)
        self.merges = {}
        self.vocab = self._build_initial_vocab()
        self._train()

    def _build_initial_vocab(self):
        vocab = set()
        for text in self.corpus:
            words = text.split()
            for word in words:
                vocab.update(list(word + '</w>'))
        print(f"Initial vocab size: {len(vocab)}")
        return {char: char for char in vocab}

    def _get_pair_frequencies(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        new_token = ''.join(pair)
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            symbols = word.split()
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_freqs[' '.join(new_symbols)] = freq
        return new_word_freqs

    def _train(self):
        word_freqs = Counter()
        for text in self.corpus:
            words = [w for w in text.split() if w]  # Skip empty words
            for word in words:
                word_freqs[' '.join(list(word) + ['</w>'])] += 1

        if not word_freqs:
            print("Error: No valid words found in corpus.")
            return

        current_vocab_size = len(self.vocab)
        merge_id = 0
        while current_vocab_size < self.vocab_size:
            pairs = self._get_pair_frequencies(word_freqs)
            if not pairs:
                print("No more pairs to merge.")
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges[merge_id] = best_pair
            self.vocab[''.join(best_pair)] = ''.join(best_pair)
            current_vocab_size += 1
            merge_id += 1
        print(f"Final vocab size: {len(self.vocab)}")
        print(f"Sample vocab: {list(self.vocab.keys())[:10]}")

    def tokenize(self, text):
        words = [w for w in text.split() if w]  # Skip empty words
        tokenized = []
        for word in words:
            symbols = list(word) + ['</w>']
            word_tokens = []
            i = 0
            while i < len(symbols):
                # Try to find the longest matching merge
                matched = False
                for j in range(len(symbols) - i, 0, -1):
                    candidate = ''.join(symbols[i:i+j])
                    if candidate in self.vocab:
                        word_tokens.append(candidate)
                        i += j
                        matched = True
                        break
                if not matched:
                    word_tokens.append(symbols[i])
                    i += 1
            tokenized.extend(word_tokens)
        print(f"Tokenized '{text[:30]}...': {word_tokens[:10]}")
        return tokenized

# Step 3: Embedding Model
class SkipGramEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context, noise):
        emb_target = self.embeddings(target).unsqueeze(1)
        emb_context = self.output(context)
        emb_noise = self.output(noise)
        pos_scores = torch.sum(emb_target * emb_context, dim=2)
        neg_scores = torch.sum(emb_target * emb_noise, dim=2)
        return pos_scores, neg_scores

# Step 4: Train Embeddings
def train_embeddings(tokenized_corpus, vocab, embed_dim=50, window_size=2, num_noise=5, epochs=10, batch_size=32, lr=0.01):
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    vocab_size = len(vocab)
    print(f"Vocabulary size for embeddings: {vocab_size}")

    pairs = []
    for tokens in tokenized_corpus:
        if not tokens:
            continue
        for i, target in enumerate(tokens):
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if i != j and tokens[j] in token_to_idx:
                    pairs.append((token_to_idx[target], token_to_idx[tokens[j]]))
    
    print(f"Generated {len(pairs)} context-target pairs.")

    if len(pairs) == 0:
        print("Error: No valid context-target pairs generated. Check corpus size or tokenization.")
        return None, None

    model = SkipGramEmbeddings(vocab_size, embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0
        batch_count = 0
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start + batch_size]
            if not batch_pairs:
                continue
            targets = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long)
            contexts = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long).unsqueeze(1)
            noise = torch.randint(0, vocab_size, (len(batch_pairs), num_noise))
            pos_scores, neg_scores = model(targets, contexts, noise)
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            loss = criterion(pos_scores, pos_labels) + criterion(neg_scores, neg_labels)
            total_loss += loss.item()
            batch_count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model.embeddings.weight.data.numpy(), token_to_idx

# Example Usage
if __name__ == "__main__":
    # Option 1: Use a local file
    # file_path = "play.txt"
    # corpus = load_corpus(file_path=file_path)

    # Option 2: Use a URL
    url = "https://www.gutenberg.org/files/1513/1513-0.txt"  # Romeo and Juliet
    corpus = load_corpus(url=url)

    # Option 3: Use sample text (uncomment if needed)
    # corpus = load_corpus()

    if not corpus:
        print("Error: Empty corpus. Please provide a valid file or URL.")
        exit()

    # Train BPE tokenizer
    tokenizer = BPETokenizer(corpus, vocab_size=500)
    tokenized_corpus = [tokenizer.tokenize(text) for text in corpus]
    
    # Check tokenized corpus
    print(f"Tokenized corpus sample: {tokenized_corpus[:2]}")

    # Build vocab
    vocab = list(set(token for tokens in tokenized_corpus for token in tokens if token))
    
    if not vocab:
        print("Error: Empty vocabulary after tokenization.")
        exit()

    # Train embeddings
    embeddings, token_to_idx = train_embeddings(tokenized_corpus, vocab, embed_dim=20, epochs=5)

    if embeddings is not None and token_to_idx is not None:
        sample_token = "romeo" if "romeo" in token_to_idx else list(token_to_idx.keys())[0]
        print(f"Embedding for '{sample_token}':\n{embeddings[token_to_idx[sample_token]]}")