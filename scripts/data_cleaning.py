import os
import sys

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.clean_data_pipeline import run_cleaning_pipeline

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "..", "data", "raw", "laptop_features.csv")
    output_file = os.path.join(current_dir, "..", "data", "raw", "clean_laptop_features.csv")
    run_cleaning_pipeline(input_file, output_file)
