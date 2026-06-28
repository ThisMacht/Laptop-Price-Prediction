import os
import yaml
import pandas as pd
import time
from src.data.chotot_collect import get_product_links, get_product_details

def load_config():
    config_path = "config/chotot_collection.yml"
    if not os.path.exists(config_path):
        os.makedirs("config", exist_ok=True)
        default_config = {
            'pipeline': {
                'total_pages': 1, # Thử nghiệm với 1 trang trước
                'data_output': "data/raw/chotot_laptop_data.csv",
                'delay_between_pages': 2,
                'delay_between_details': 1
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    pipeline_cfg = config['pipeline']
    
    total_pages = pipeline_cfg.get('total_pages', 1)
    all_details = []
    
    print(f"--- Bắt đầu thu thập link sản phẩm (Dùng class cwv3xk0) ---")
    all_links = []
    for page in range(1, total_pages + 1):
        print(f"Đang lấy link từ trang {page}...")
        links = get_product_links(page)
        print(f"Tìm thấy {len(links)} link sản phẩm.")
        all_links.extend(links)
        time.sleep(pipeline_cfg.get('delay_between_pages', 3))
    
    all_links = list(set(all_links)) # Unique links
    print(f"Tổng cộng có {len(all_links)} link sản phẩm cần cào chi tiết.")
    
    print(f"\n--- Bắt đầu thu thập chi tiết từng sản phẩm ---")
    for i, url in enumerate(all_links):
        print(f"[{i+1}/{len(all_links)}] Đang cào: {url}")
        details = get_product_details(url)
        if details:
            all_details.append(details)
        
        # Delay để tránh bị block
        time.sleep(pipeline_cfg.get('delay_between_details', 2))
        
        # Lưu checkpoint mỗi 10 sản phẩm
        if (i + 1) % 10 == 0:
            pd.DataFrame(all_details).to_csv(pipeline_cfg.get('data_output'), index=False, encoding='utf-8-sig')
            print(f"--- Đã lưu checkpoint {len(all_details)} sản phẩm ---")

    if all_details:
        df = pd.DataFrame(all_details)
        output_path = pipeline_cfg.get('data_output', "data/raw/chotot_laptop_data.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n--- Hoàn tất! Đã lưu tổng cộng {len(all_details)} laptop vào {output_path} ---")
    else:
        print("Không có dữ liệu chi tiết nào được thu thập.")

if __name__ == "__main__":
    main()
