import os
import pandas as pd
from itertools import combinations
from collections import Counter
import inflect

# === Initialize inflect engine for singularization ===
inflect_engine = inflect.engine()

def singularize_entity(entity):
    """Convert an entity to its singular form if applicable."""
    return inflect_engine.singular_noun(entity) if inflect_engine.singular_noun(entity) else entity

def compute_entity_cooccurrences(corpus_file, output_directory, chunksize=1000, max_entities_per_article=100):
    """
    Compute entity co-occurrences while maintaining correct PUBMED_ID mapping.

    Args:
        corpus_file (str): Path to the corpus file containing extracted entities.
        output_directory (str): Directory where the co-occurrence file will be saved.
        chunksize (int): Number of rows processed at a time to optimize memory usage.

    Returns:
        str: Path to the saved co-occurrence file.
    """
    try:
        print(f"[INFO] Processing corpus file: {corpus_file}")

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        cooccurrence_file = os.path.join(output_directory, "entity_cooccurrences.csv")

        # Global counter for entity occurrences
        entity_occurrence_map = Counter()
        total_articles = 0
        cooccurrence_data = []

        # Process the corpus in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(corpus_file, chunksize=chunksize)):
            print(f"[INFO] Processing chunk {chunk_idx+1}...")

            # Standardize column names
            chunk.columns = chunk.columns.str.strip().str.upper()
            required_columns = {"PUBMED_ID", "ENTITY", "LABEL"}
            missing_columns = required_columns - set(chunk.columns)

            # Skip chunks with missing columns
            if missing_columns:
                print(f"[ERROR] Missing columns in chunk {chunk_idx+1}: {missing_columns}. Skipping...")
                continue

            # Normalize entity and label names
            chunk["ENTITY"] = chunk["ENTITY"].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "UNKNOWN")
            chunk["LABEL"] = chunk["LABEL"].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "UNKNOWN")

            # Track entity occurrences globally
            for entity in chunk["ENTITY"]:
                entity_occurrence_map[entity] += 1  

            # Process co-occurrences per article
            for pubmed_id, group in chunk.groupby("PUBMED_ID"):
                total_articles += 1
                entities_labels = [(str(entity).lower(), str(label).lower()) for entity, label in zip(group["ENTITY"], group["LABEL"])]

                # Skip articles with too few entities
                if len(entities_labels) < 2:
                    continue

                # Limit the number of entities per article
                if len(entities_labels) > max_entities_per_article:
                    entities_labels = entities_labels[:max_entities_per_article]

                # Compute co-occurrences within the same document
                local_cooccurrence_counter = Counter()
                for (source, source_type), (target, target_type) in combinations(entities_labels, 2):
                    if source > target:
                        source, source_type, target, target_type = target, target_type, source, source_type
                    
                    # Local count per article
                    local_cooccurrence_counter[(source, source_type, target, target_type)] += 1

                # Store co-occurrences for this article
                for (src, stype, tgt, ttype), count in local_cooccurrence_counter.items():
                    cooccurrence_data.append({
                        "PUBMED_ID": pubmed_id,
                        "SOURCE": src,
                        "SOURCE_TYPE": stype,
                        "TARGET": tgt,
                        "TARGET_TYPE": ttype,
                        "COOCCURRENCE": count,
                        "SOURCE_OCCURRENCE": entity_occurrence_map.get(src, 0),
                        "TARGET_OCCURRENCE": entity_occurrence_map.get(tgt, 0)
                    })

        print(f"[INFO] Processed {total_articles} articles.")

        # Save co-occurrences to CSV
        if cooccurrence_data:
            cooccurrence_df = pd.DataFrame(cooccurrence_data)

            # Ensure PUBMED_ID is the first column
            column_order = ["PUBMED_ID", "SOURCE", "SOURCE_TYPE", "TARGET", "TARGET_TYPE", "COOCCURRENCE", "SOURCE_OCCURRENCE", "TARGET_OCCURRENCE"]
            cooccurrence_df = cooccurrence_df[column_order]

            cooccurrence_df.to_csv(cooccurrence_file, index=False)
            print(f"[SUCCESS] Co-occurrences saved to: {cooccurrence_file}")

            # Display the first few rows for verification
            print("[INFO] Checking final structure of co-occurrence file:")
            print(cooccurrence_df.head())

        return cooccurrence_file

    except Exception as e:
        print(f"[ERROR] Failed to compute co-occurrences: {e}")
