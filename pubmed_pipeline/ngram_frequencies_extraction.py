import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tqdm import tqdm

def extract_ngrams_from_corpus(corpus_file, output_directory, ngram_range=(1, 3), batch_size=5000):
    """
    Extracts N-grams from the TEXT column of the corpus using CountVectorizer.

    Args:
        corpus_file (str): Path to the corpus file containing the TEXT column.
        output_directory (str): Directory where the N-gram file will be saved.
        ngram_range (tuple): Min and max range for N-grams (default: unigrams to trigrams).
        batch_size (int): Number of rows processed at a time (to reduce memory usage).

    Returns:
        str: Path to the saved N-grams file.
    """
    try:
        print("[INFO] Loading corpus file for N-gram extraction...")

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, "ngrams_extracted.csv")

        # Read corpus to check column structure
        first_chunk = pd.read_csv(corpus_file, nrows=5)
        first_chunk.columns = first_chunk.columns.str.strip().str.upper()

        if "TEXT" not in first_chunk.columns:
            print(f"[ERROR] Column 'TEXT' not found in corpus file: {corpus_file}. Aborting...")
            return None

        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
        ngrams_count = Counter()

        print(f"[INFO] Processing N-grams from TEXT column in batches of {batch_size}...")

        for chunk_idx, chunk in enumerate(pd.read_csv(corpus_file, chunksize=batch_size)):
            print(f"[INFO] Processing chunk {chunk_idx + 1}...")

            # Normalize column names
            chunk.columns = chunk.columns.str.strip().str.upper()

            # Ensure TEXT column exists
            if "TEXT" not in chunk.columns:
                print(f"[ERROR] Missing 'TEXT' column in chunk {chunk_idx+1}. Skipping...")
                continue

            # Drop NaN values and empty rows
            chunk["TEXT"] = chunk["TEXT"].fillna("").astype(str)
            texts = chunk["TEXT"][chunk["TEXT"].str.strip() != ""]

            if texts.empty:
                print(f"[WARNING] No valid texts in chunk {chunk_idx+1}. Skipping...")
                continue

            # Extract N-grams
            ngrams = vectorizer.fit_transform(texts)
            ngrams_list = vectorizer.get_feature_names_out()
            ngram_counts = ngrams.toarray().sum(axis=0)

            for ngram, count in zip(ngrams_list, ngram_counts):
                ngrams_count[ngram] += count  # Accumulate across chunks

        if not ngrams_count:
            print("[ERROR] No valid N-grams were extracted. Check input data.")
            return None

        # Convert dictionary to DataFrame
        ngram_table = pd.DataFrame({
            "N": [len(ngram.split()) for ngram in ngrams_count.keys()],
            "Ngram": list(ngrams_count.keys()),
            "Count": list(ngrams_count.values())
        }).sort_values(by="Count", ascending=False)

        # Save to CSV
        ngram_table.to_csv(output_file, index=False)
        print(f"[SUCCESS] N-gram extraction completed and saved to {output_file}")

        # Display first few rows for verification
        print("[INFO] First 5 N-grams extracted:")
        print(ngram_table.head())

        return output_file

    except Exception as e:
        print(f"[ERROR] Failed to extract N-grams: {e}")
        return None
