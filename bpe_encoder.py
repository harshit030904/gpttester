from collections import Counter, deque
from functools import lru_cache
import json
import os
import urllib.request
import re

# Function to download file if not present
def download_file_if_absent(url, filename, search_dirs="."):
    for directory in [search_dirs]:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            print(f"{filename} already exists in {file_path}")
            return file_path

    target_path = os.path.join(search_dirs, filename)
    try:
        with urllib.request.urlopen(url) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Downloaded {filename} to {target_path}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")
    return target_path

# Download GPT-2 vocab and merges files
files_to_download = {
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json"
}

paths = {}
for url, filename in files_to_download.items():
    paths[filename] = download_file_if_absent(url, filename, search_dirs=".")

# Download Hamlet dataset for testing
hamlet_path = download_file_if_absent(
    url="https://www.gutenberg.org/cache/epub/1524/pg1524.txt",
    filename="hamlet.txt",
    search_dirs="."
)

# Read a small portion of Hamlet for testing
with open(hamlet_path, "r", encoding="utf-8") as f:
    text = f.read(1000)  # Limited to 1000 chars to keep it manageable

class BPETokenizerSimple:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.bpe_ranks = {}

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character
        if "\n" not in self.inverse_vocab:
            fallback_token = next((token for token in ["<|endoftext|>", "Ġ", ""] if token in self.inverse_vocab), None)
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")
            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # Load GPT-2 merges
        self.bpe_ranks = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
            rank = 0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                    else:
                        print(f"Skipping pair {pair} as one token is not in the vocabulary.")

    def encode(self, text, allowed_special=None):
        token_ids = []
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )
            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None))
                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
            text = text[last_index:]
            disallowed = [
                tok for tok in self.inverse_vocab
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]
            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")

        # Modified: Replace spaces with Ġ to match GPT-2 vocab
        text = text.replace(" ", "Ġ")
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split("Ġ")
            for j, word in enumerate(words):
                if word:  # Skip empty words
                    if j == 0 and i > 0:
                        tokens.append("Ġ" + word)
                    elif j == 0:
                        tokens.append(word)
                    else:
                        tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def tokenize_with_bpe(self, token):
        # Handle space explicitly
        token = token.replace(" ", "Ġ")  # Ensure spaces are replaced with Ġ
        token_ids = [self.inverse_vocab.get(char, self.inverse_vocab.get("Ġ", None)) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        symbols = [self.vocab[id_num] for id_num in token_ids]
        while True:
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p
            if bigram is None or bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]  # Replace Ġ with space
            else:
                decoded_string += token
        return decoded_string

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

# Initialize and load GPT-2 tokenizer
tokenizer_gpt2 = BPETokenizerSimple()
tokenizer_gpt2.load_vocab_and_merges_from_openai(
    vocab_path=paths["encoder.json"],
    bpe_merges_path=paths["vocab.bpe"]
)

# Test 1: Generic text
input_text = "This is some text"
token_ids = tokenizer_gpt2.encode(input_text)
print("Test 1 - Generic Text")
print("Input:", input_text)
print("Token IDs:", token_ids)
print("Decoded:", tokenizer_gpt2.decode(token_ids))

# Test 2: Hamlet-inspired text with special token
input_text = "To be or not to be, that is the question.<|endoftext|>"
token_ids = tokenizer_gpt2.encode(input_text, allowed_special={"<|endoftext|>"})
print("\nTest 2 - Hamlet Text with Special Token")
print("Input:", input_text)
print("Token IDs:", token_ids)
print("Decoded:", tokenizer_gpt2.decode(token_ids))

# Test 3: Small portion of Hamlet text
hamlet_sample = text[:100]  # First 100 chars of Hamlet
token_ids = tokenizer_gpt2.encode(hamlet_sample)
print("\nTest 3 - Hamlet Sample")
print("Input:", hamlet_sample)
print("Token IDs:", token_ids)
print("Decoded:", tokenizer_gpt2.decode(token_ids))

# Test 4: Check vocab size
print("\nVocabulary size:", len(tokenizer_gpt2.vocab))