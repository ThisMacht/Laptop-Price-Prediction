import sys
import os
import argparse
import yaml

# Add the project root to the python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pipeline.collection_data_pipeline import DataCollectionPipeline

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

def main():
    default_config_path = os.path.join(project_root, "config", "websosanh_collection.yml")
    
    parser = argparse.ArgumentParser(description="Run Laptop Data Collection Pipeline")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to the YAML configuration file")
    
    # Optional overrides from CLI
    parser.add_argument("--start-page", type=int, default=None, help="Starting page to scrape (overrides config)")
    parser.add_argument("--end-page", type=int, default=None, help="Ending page to scrape (overrides config)")
    parser.add_argument("--links-file", type=str, default=None, help="File to save/read laptop links (overrides config)")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV file for scraped data (overrides config)")
    parser.add_argument("--step", type=str, choices=["all", "links", "data"], default="all", help="Which step to run")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize Pipeline
    pipeline = DataCollectionPipeline(
        config=config,
        start_page=args.start_page,
        end_page=args.end_page,
        links_output=args.links_file,
        data_output=args.output_csv
    )
    
    if args.step == "all":
        pipeline.run_full_pipeline()
    elif args.step == "links":
        pipeline.run_url_extraction()
    elif args.step == "data":
        pipeline.run_data_extraction()

if __name__ == "__main__":
    main()
