import math
from collections import Counter


def extract_token_metafeatures(text: str):
    # 1. Basic Tokenization (For a hackathon MVP, splitting by whitespace is fast and effective)
    tokens = text.lower().split()
    total_tokens = len(tokens)

    if total_tokens == 0:
        return None

    # 2. Calculate Frequencies
    token_counts = Counter(tokens)
    unique_tokens = len(token_counts)

    # --- METAFEATURE 1: Type-Token Ratio (TTR) ---
    ttr = unique_tokens / total_tokens

    # --- METAFEATURE 2: Hapax Legomena Ratio ---
    # Count how many tokens appear exactly 1 time
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    hapax_ratio = hapax_count / unique_tokens

    # --- METAFEATURE 3: Zipf's Alpha (Approximation) ---
    # We estimate alpha by comparing the frequency of the 1st and 10th most common words.
    # Zipf's formula: f(r) = c / (r^alpha) => alpha = log(f(1)/f(10)) / log(10)
    sorted_counts = [count for token, count in token_counts.most_common()]

    if len(sorted_counts) >= 10 and sorted_counts[9] > 0:
        freq_rank_1 = sorted_counts[0]
        freq_rank_10 = sorted_counts[9]
        zipf_alpha = math.log(freq_rank_1 / freq_rank_10) / math.log(10)
    else:
        zipf_alpha = 1.0  # Default fallback if dataset is too small

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr_score": round(ttr, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "zipf_alpha": round(zipf_alpha, 4),
    }


# --- Test the Engine ---
if __name__ == "__main__":
    # Mocking a highly repetitive dataset (Low TTR, Low Hapax)
    repetitive_text = "the cat sat on the mat the cat is orange the mat is brown" * 100

    # Mocking a diverse dataset (High TTR, High Hapax)
    diverse_text = "artificial intelligence leverages neural networks to optimize complex thermodynamic routing algorithms globally"

    print("Repetitive Dataset Metafeatures:")
    print(extract_token_metafeatures(repetitive_text))

    print("\nDiverse Dataset Metafeatures:")
    print(extract_token_metafeatures(diverse_text))
