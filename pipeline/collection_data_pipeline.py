import time
import os
import json
import pandas as pd
from src.data.collect import (
    scrape_websosanh_api,
    find_detail_urls,
    fetch_laptop_html,
    extract_table_specifications,
    fetch_prices_and_shops
)

class DataCollectionPipeline:
    def __init__(self, config=None, start_page=None, end_page=None, links_output=None, data_output=None):
        self.config = config or {}
        pipeline_config = self.config.get('pipeline', {})
        api_config = self.config.get('api', {})

        # Priority: explicit kwargs -> config -> defaults
        self.start_page = start_page if start_page is not None else pipeline_config.get('start_page', 1)
        self.end_page = end_page if end_page is not None else pipeline_config.get('end_page', 10)
        self.links_output = links_output if links_output is not None else pipeline_config.get('links_output', "websosanh_product_links.txt")
        self.data_output = data_output if data_output is not None else pipeline_config.get('data_output', "laptop_features.csv")
        self.delay = pipeline_config.get('delay_between_requests', 2)
        self.chunk_size = pipeline_config.get('chunk_size', 20)
        self.brands_to_crawl = pipeline_config.get('brands', 'all')
        
        self.base_url = api_config.get('base_url', "https://websosanh.vn")

        # Load brand mapping
        self.brand_mapping = {}
        mapping_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'encoders', 'brand_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.brand_mapping = json.load(f)

    def run_url_extraction(self):
        unique_urls = set()
        print(f"Starting to extract URLs from page {self.start_page} to {self.end_page}...")
        
        if not self.brand_mapping:
            print("Warning: brand_mapping.json not found or empty.")
            target_brands = {'all': None}
        else:
            if self.brands_to_crawl == 'all':
                target_brands = self.brand_mapping
            else:
                target_brands = {b: self.brand_mapping[b] for b in self.brands_to_crawl if b in self.brand_mapping}
                
        for brand, cat_id in target_brands.items():
            print(f"\n--- Crawling brand: {brand} (Category ID: {cat_id}) ---")
            for page in range(self.start_page, self.end_page + 1):
                print(f"Processing {brand} page {page} for links...")
                result_data = scrape_websosanh_api(page_index=page, config=self.config, brand=brand, category_id=cat_id)
                
                if result_data:
                    relative_urls = find_detail_urls(result_data)
                    
                    for rel_url in relative_urls:
                        absolute_url = self.base_url + rel_url
                        unique_urls.add(absolute_url)
                        
                # Pause to avoid rate limits
                time.sleep(self.delay)
            
        print(f"Extraction finished. Found {len(unique_urls)} unique URLs.")
        
        if unique_urls:
            # Make sure directory exists before saving
            os.makedirs(os.path.dirname(os.path.abspath(self.links_output)), exist_ok=True)
            with open(self.links_output, "w", encoding="utf-8") as file:
                for url in unique_urls:
                    file.write(url + "\n")
            print(f"All URLs successfully saved to {self.links_output}")
            
        return list(unique_urls)

    def run_data_extraction(self, urls=None):
        if not urls:
            if not os.path.exists(self.links_output):
                print(f"File {self.links_output} not found. Running URL extraction first...")
                urls = self.run_url_extraction()
            else:
                with open(self.links_output, "r", encoding="utf-8") as f:
                    urls = [line.strip() for line in f if line.strip()]
                    
        chunk = []
        total_extracted = 0
        
        print(f"Starting data extraction for {len(urls)} URLs...")
        for i, url in enumerate(urls):
            print(f"Processing URL {i+1}/{len(urls)}: {url}")
            raw_html = fetch_laptop_html(url, config=self.config)
            
            if raw_html:
                specs = extract_table_specifications(raw_html)
                if specs:
                    specs['source_url'] = url
                    
                    try:
                        parts = url.split('/')
                        product_id = [p for p in parts if p.isdigit()][-1]
                        
                        price_info = fetch_prices_and_shops(product_id, config=self.config)
                        specs.update(price_info)
                    except Exception as e:
                        print(f"Failed to fetch price for {url}: {e}")
                        
                    chunk.append(specs)
                    total_extracted += 1
                    
            if len(chunk) >= self.chunk_size:
                self._save_chunk(chunk)
                chunk = []
                print(f"--- Checkpoint: Saved {total_extracted} laptops so far... ---")
            
            time.sleep(self.delay)
            
        if chunk:
            self._save_chunk(chunk)
            
        print(f"Data extraction finished. Gathered data for {total_extracted} laptops.")
        print(f"Data successfully saved to {self.data_output}")

    def _save_chunk(self, chunk):
        if not chunk:
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(self.data_output)), exist_ok=True)
        new_df = pd.DataFrame(chunk)
        if os.path.exists(self.data_output):
            existing_df = pd.read_csv(self.data_output)
            # Combine to ensure headers are aligned properly across chunks with varying layouts
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(self.data_output, index=False, encoding='utf-8')
        else:
            new_df.to_csv(self.data_output, index=False, encoding='utf-8')

    def run_full_pipeline(self):
        print("--- Starting Full Data Collection Pipeline ---")
        urls = self.run_url_extraction()
        self.run_data_extraction(urls=urls)
        print("--- Pipeline Execution Completed ---")
