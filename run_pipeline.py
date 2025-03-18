import argparse
import os
from datetime import datetime
from run_full_pipeline import run_full_pipeline  # Import the modified function

# Command-line argument parser
parser = argparse.ArgumentParser(description="Run the PubMed Information Extraction Pipeline")

# Required arguments
parser.add_argument("--search_term", type=str, required=True, help="Search term for PubMed articles")
parser.add_argument("--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format")
parser.add_argument("--end_date", type=str, required=True, help="End date in YYYY-MM-DD format")

# Optional argument: Base directory
parser.add_argument("--base_dir", type=str, required=True, help="Base directory where Input and Output folders will be created")

# Parse arguments
args = parser.parse_args()

# Convert string dates to datetime format
try:
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
except ValueError:
    print("[ERROR] Invalid date format. Please use YYYY-MM-DD.")
    exit(1)

# Define directories
BASE_DIR = args.base_dir
input_dir = os.path.join(BASE_DIR, "Input")
output_dir = os.path.join(BASE_DIR, "Output")

# Ensure directories exist
for directory in [input_dir, output_dir]:
    os.makedirs(directory, exist_ok=True)

print(f"[INFO] Using base directory: {BASE_DIR}")
print(f"[INFO] Input directory: {input_dir}")
print(f"[INFO] Output directory: {output_dir}")

# Run the pipeline with user-defined parameters
run_full_pipeline(args.search_term, start_date, end_date, BASE_DIR)
