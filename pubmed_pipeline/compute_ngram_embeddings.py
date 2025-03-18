import os
import spacy
import pandas as pd
from tqdm import tqdm

def compute_ngram_embeddings(ngram_file, output_directory, model="en_core_web_sm", chunksize=1000):
    """
    Generates embeddings for N-grams using spaCy in an optimized manner.

    Args:
        ngram_file (str): Path to the file containing N-grams.
        output_directory (str): Directory where embeddings will be saved.
        model (str): spaCy model to use for embeddings.
        chunksize (int): Number of rows processed per batch to optimize memory usage.

    Returns:
        str: Path to the saved embeddings file.
    """
    try:
        print("[INFO] Checking N-gram file before computing embeddings...")

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Define the output file
        output_file = os.path.join(output_directory, "ngram_embeddings.csv")

        # Check if the input file exists and is not empty
        if not os.path.exists(ngram_file) or os.stat(ngram_file).st_size == 0:
            print(f"[ERROR] N-gram file {ngram_file} is missing or empty. Exiting...")
            return None

        # Load spaCy model
        print("[INFO] Loading spaCy model...")
        try:
            nlp = spacy.load(model)
            print("[SUCCESS] spaCy model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load spaCy model '{model}': {e}")
            return None

        first_write = True
        for chunk_idx, chunk in enumerate(pd.read_csv(ngram_file, chunksize=chunksize)):
            print(f"[INFO] Processing chunk {chunk_idx + 1}...")

            # Ensure "Ngram" column exists
            if "Ngram" not in chunk.columns:
                print(f"[ERROR] Missing 'Ngram' column in chunk {chunk_idx+1}. Skipping...")
                continue

            # Compute embeddings using spaCy pipe for efficiency
            embeddings = []
            for doc in tqdm(nlp.pipe(chunk["Ngram"].astype(str)), total=len(chunk), desc="Generating embeddings", leave=False):
                embeddings.append(doc.vector)

            # Convert embeddings to DataFrame
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.insert(0, "Ngram", chunk["Ngram"].values)

            # Append to CSV file progressively
            if os.path.exists(output_file) and not first_write:
                embeddings_df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                embeddings_df.to_csv(output_file, mode='w', header=True, index=False)
                first_write = False

            print(f"[SUCCESS] Chunk {chunk_idx+1} processed and saved.")

        print(f"[SUCCESS] All N-gram embeddings saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"[ERROR] Failed to compute embeddings: {e}")
        return None
