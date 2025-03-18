import os
import spacy
import pandas as pd
from tqdm import tqdm
import gc
import time

# Load SpaCy model once (global variable)
nlp_bionlp = spacy.load("en_ner_bionlp13cg_md", disable=["parser", "tagger", "lemmatizer"])
print("[INFO] SpaCy model loaded successfully.")

def process_compiled_file_with_bionlp(file_path, output_directory, batch_size=5000):
    """
    Processes a PubMed articles CSV file using en_ner_bionlp13cg_md to extract named entities efficiently in batch mode.
    
    Args:
        file_path (str): Path to the input CSV file containing PubMed articles.
        output_directory (str): Directory where extracted entities will be saved.
        batch_size (int): Number of documents processed per batch to optimize performance.
    
    Returns:
        None: Saves extracted entities as a CSV file.
    """

    # Step 1: Load the file in chunks
    try:
        chunks = pd.read_csv(file_path, chunksize=1000)  # Process data in chunks
        print(f"[INFO] Processing file: {file_path}")
    except Exception as e:
        print(f"[ERROR] Error loading file {file_path}: {e}")
        return

    # Step 2: Define the output file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_directory, f"{base_name}_entities.csv")

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Step 3: Process data in chunks
    total_start_time = time.time()
    for chunk_idx, df in enumerate(chunks):
        print(f"[INFO] Processing chunk {chunk_idx+1}...")

        # Standardize column names
        df.columns = df.columns.str.strip().str.upper()

        # Verify required columns
        required_columns = {'PUBMED_ID', 'TITLE', 'ABSTRACT', 'PUBLICATION DATE'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"[ERROR] Missing columns in file {file_path}: {missing_columns}. Skipping chunk {chunk_idx+1}.")
            continue

        # Remove empty rows in TITLE and ABSTRACT
        df = df.dropna(subset=['TITLE', 'ABSTRACT'])
        if df.empty:
            print(f"[WARNING] Chunk {chunk_idx+1} contains no valid rows. Skipping...")
            continue

        # Step 4: Batch processing
        results = []
        texts = df['TITLE'] + " " + df['ABSTRACT']
        pubmed_ids = df['PUBMED_ID'].tolist()
        titles = df['TITLE'].tolist()
        abstracts = df['ABSTRACT'].tolist()
        publication_dates = df['PUBLICATION DATE'].tolist()

        batch_start_time = time.time()

        # Process each text using SpaCy's pipeline
        for doc, pubmed_id, title, abstract, pub_date in tqdm(
            zip(nlp_bionlp.pipe(texts, batch_size=batch_size, disable=["parser", "tagger", "lemmatizer"]),
                pubmed_ids, titles, abstracts, publication_dates),
            total=len(texts), desc="Processing Batches"
        ):
            # Extract named entities
            for ent in doc.ents:
                results.append({
                    'PUBMED_ID': pubmed_id,
                    'TITLE': title,
                    'ABSTRACT': abstract,
                    'PUBLICATION DATE': pub_date,
                    'ENTITY': ent.text.strip().lower(),  # Ensure entity text is properly formatted
                    'LABEL': ent.label_  # Assign entity label
                })

        batch_end_time = time.time()
        print(f"[INFO] Chunk {chunk_idx+1} processed in {batch_end_time - batch_start_time:.2f} seconds")

        # Step 5: Save extracted entities (append mode)
        if results:
            entities_df = pd.DataFrame(results)

            # Final column verification: Ensure all expected columns exist
            expected_columns = ['PUBMED_ID', 'TITLE', 'ABSTRACT', 'PUBLICATION DATE', 'ENTITY', 'LABEL']
            missing_columns = set(expected_columns) - set(entities_df.columns)
            if missing_columns:
                print(f"[ERROR] Missing columns in output file: {missing_columns}. Fixing...")
                for col in missing_columns:
                    entities_df[col] = "UNKNOWN"

            # Save results to CSV
            entities_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            print(f"[SUCCESS] Saved {len(entities_df)} entities from chunk {chunk_idx+1}")

        # Free memory after processing
        del results, df
        gc.collect()

    total_end_time = time.time()
    print(f"[SUCCESS] Total time for processing {file_path}: {total_end_time - total_start_time:.2f} seconds")
