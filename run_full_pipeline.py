import os
import argparse
import pandas as pd
from datetime import datetime
from fetch_pubmed_articles import fetch_pubmed_articles_by_week
from process_bionlp import process_compiled_file_with_bionlp
from compute_entity_cooccurrences import compute_entity_cooccurrences
from compute_entity_similarities import compute_entity_similarities
from ngram_frequencies_extraction import extract_ngrams_from_corpus
from compute_ngram_embeddings import compute_ngram_embeddings
from create_corpus import create_corpus

# === Function to Sort Files Chronologically ===
def sort_by_date(file_list, prefix="pubmed_articles_", suffix=".csv"):
    """Sort filenames chronologically based on the date in the filename."""
    def extract_date(file_name):
        try:
            parts = file_name.replace(prefix, "").replace(suffix, "").split("_to_")
            return datetime.strptime(parts[0], "%Y-%m-%d")
        except Exception as e:
            print(f"[WARNING] Could not extract date from {file_name}: {e}")
            return datetime.min  # Push improperly formatted files to the start

    return sorted(file_list, key=extract_date)

# === Full Pipeline Execution ===
def run_full_pipeline(search_term, start_date, end_date, base_dir):
    """
    Run the full PubMed pipeline with user-defined parameters.

    Args:
        search_term (str): The term to search in PubMed.
        start_date (datetime): The start date.
        end_date (datetime): The end date.
        base_dir (str): The base directory where data is stored.
    """

    # === Directory Configuration ===
    input_dir = os.path.join(base_dir, "Input")
    output_dir = os.path.join(base_dir, "Output")
    entities_dir = os.path.join(output_dir, "entities")
    cooccurrences_dir = os.path.join(output_dir, "cooccurrences")
    similarities_dir = os.path.join(output_dir, "similarities")
    ngrams_dir = os.path.join(output_dir, "ngrams")
    embeddings_dir = os.path.join(output_dir, "embeddings")

    # Ensure all necessary directories exist
    for directory in [input_dir, output_dir, entities_dir, cooccurrences_dir, similarities_dir, ngrams_dir, embeddings_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"[INFO] Starting pipeline in: {base_dir}")
    print(f"[INFO] Using search term: '{search_term}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # === STEP 1: Fetching PubMed articles ===
    try:
        print("[INFO] Step 1: Fetching PubMed articles...")
        articles_files = sort_by_date([f for f in os.listdir(input_dir) if f.endswith(".csv")])

        if articles_files:
            print("[INFO] PubMed articles already fetched. Skipping step.")
        else:
            fetch_pubmed_articles_by_week(search_term, start_date, end_date, output_directory=input_dir)
            print("[SUCCESS] PubMed articles fetched successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to fetch PubMed articles: {e}")
        return

    # === STEP 2: Entity Extraction ===
    try:
        print("[INFO] Step 2: Extracting entities...")

        input_files = sort_by_date([f for f in os.listdir(input_dir) if f.endswith(".csv")])
        extracted_files = {os.path.splitext(f)[0].replace("_entities", "") for f in os.listdir(entities_dir) if f.endswith("_entities.csv")}

        missing_files = [f for f in input_files if os.path.splitext(f)[0] not in extracted_files]

        if not missing_files:
            print("[INFO] All entities already extracted. Skipping step.")
        else:
            print(f"[INFO] {len(missing_files)} files require entity extraction. Processing...")

            for file in missing_files:
                file_path = os.path.join(input_dir, file)
                print(f"[INFO] Extracting entities from: {file_path}")
                process_compiled_file_with_bionlp(file_path, entities_dir)

            print("[SUCCESS] All entities extracted successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to extract entities: {e}")
        return

    # === STEP 3: Creating Unified Corpus ===
    corpus_file = os.path.join(cooccurrences_dir, "corpus.csv")
    if os.path.exists(corpus_file):
        print("[INFO] Corpus already created. Skipping step.")
    else:
        corpus_file = create_corpus(entities_dir, corpus_file)
        if not corpus_file:
            print("[ERROR] Corpus creation failed. Stopping pipeline.")
            return

    # === STEP 4: Computing Entity Cooccurrences ===
    try:
        print("[INFO] Step 4: Computing entity cooccurrences...")
        cooccurrence_file = compute_entity_cooccurrences(corpus_file, cooccurrences_dir)
        if not cooccurrence_file:
            print("[ERROR] Failed to compute cooccurrences. Stopping pipeline.")
            return
        print("[SUCCESS] Cooccurrences calculated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to calculate cooccurrences: {e}")
        return

    # === STEP 5: Computing Entity Similarities ===
    try:
        print("[INFO] Step 5: Computing entity similarities...")
        similarity_file = compute_entity_similarities(cooccurrence_file, similarities_dir)
        if not similarity_file:
            print("[ERROR] Failed to compute similarities. Stopping pipeline.")
            return
        print("[SUCCESS] Similarities calculated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to compute similarities: {e}")
        return

    # === STEP 6: Extracting N-grams ===
    try:
        print("[INFO] Step 6: Extracting N-grams from corpus...")
        ngram_file = extract_ngrams_from_corpus(corpus_file, ngrams_dir)
        if not ngram_file:
            print("[ERROR] Failed to extract N-grams. Stopping pipeline.")
            return
        print("[SUCCESS] N-grams extracted successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to extract N-grams: {e}")
        return

    # === STEP 7: Computing N-gram Embeddings ===
    try:
        print("[INFO] Step 7: Computing N-gram embeddings...")
        embedding_file = compute_ngram_embeddings(ngram_file, embeddings_dir)
        if not embedding_file:
            print("[ERROR] Failed to compute N-gram embeddings. Stopping pipeline.")
            return
        print("[SUCCESS] N-gram embeddings calculated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to compute N-gram embeddings: {e}")
        return

    print("[SUCCESS] Full pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PubMed Information Extraction Pipeline")
    parser.add_argument("--search_term", type=str, required=True, help="Search term for PubMed articles")
    parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for storing input/output data")

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("[ERROR] Invalid date format. Please use YYYY-MM-DD.")
        exit(1)

    run_full_pipeline(args.search_term, start_date, end_date, args.base_dir)
