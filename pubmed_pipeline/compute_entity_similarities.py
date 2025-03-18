import os
import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# Load BioBERT model once to avoid redundant computations
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)

def get_embedding(entity, cache):
    """
    Retrieves or computes the embedding of an entity.
    """
    entity = entity.lower().strip()
    if entity in cache:
        return cache[entity]

    inputs = tokenizer(entity, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).cpu()
    cache[entity] = embedding  # Save in cache
    return embedding

def compute_entity_similarities(cooccurrence_file, output_directory, chunksize=1000, cache_file="entity_embeddings.pkl"):
    """
    Computes entity similarities based on co-occurrences using BioBERT embeddings.
    """
    try:
        print("[INFO] Checking if cooccurrence file exists and has valid columns...")

        os.makedirs(output_directory, exist_ok=True)
        similarity_file = os.path.join(output_directory, "entity_similarities.csv")

        if not os.path.exists(cooccurrence_file) or os.stat(cooccurrence_file).st_size == 0:
            print("[ERROR] Cooccurrence file is missing or empty. Similarity computation aborted.")
            return None

        first_chunk = pd.read_csv(cooccurrence_file, nrows=5)
        required_columns = {"PUBMED_ID", "SOURCE", "TARGET", "COOCCURRENCE", "SOURCE_OCCURRENCE", "TARGET_OCCURRENCE"}
        missing_columns = required_columns - set(first_chunk.columns)
        if missing_columns:
            print(f"[ERROR] Missing columns in cooccurrence file: {missing_columns}. Similarity computation aborted.")
            return None

        # Load or initialize entity embedding cache
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                entity_embeddings = pickle.load(f)
        else:
            entity_embeddings = {}

        results = []
        for chunk_idx, chunk in enumerate(pd.read_csv(cooccurrence_file, chunksize=chunksize)):
            print(f"[INFO] Processing chunk {chunk_idx+1}...")

            chunk.columns = chunk.columns.str.strip().str.upper()

            for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc="Calculating similarities", leave=False):
                try:
                    source_embedding = get_embedding(row["SOURCE"], entity_embeddings)
                    target_embedding = get_embedding(row["TARGET"], entity_embeddings)
                    similarity = cosine_similarity(source_embedding, target_embedding).item()

                    results.append({
                        "PUBMED_ID": row["PUBMED_ID"],
                        "SOURCE": row["SOURCE"],
                        "TARGET": row["TARGET"],
                        "COOCCURRENCE": row["COOCCURRENCE"],
                        "SOURCE_OCCURRENCE": row["SOURCE_OCCURRENCE"],
                        "TARGET_OCCURRENCE": row["TARGET_OCCURRENCE"],
                        "SIMILARITY": similarity
                    })
                except Exception as e:
                    print(f"[WARNING] Error processing row {row['SOURCE']}-{row['TARGET']}: {e}")

        # Convert results to DataFrame and save
        similarity_df = pd.DataFrame(results)
        similarity_df.to_csv(similarity_file, index=False)

        print(f"[SUCCESS] Similarities saved to: {similarity_file}")
        return similarity_file

    except Exception as e:
        print(f"[ERROR] Failed to compute similarities: {e}")
