import requests
from bs4 import BeautifulSoup

def scrape_websosanh_api(page_index, config=None, brand=None, category_id=None):
    if config is None:
        config = {}
        
    api_config = config.get('api', {})
    payload_config = api_config.get('payload', {})
    headers_config = api_config.get('headers', {})
    
    url = api_config.get('search_endpoint', "https://websosanh.vn/search-api/get-search-product")
    timeout = api_config.get('timeout_seconds', 10)
    base_url = api_config.get('base_url', "https://websosanh.vn")
    
    cat_id = category_id if category_id else payload_config.get('category_id', 18)
    
    if brand and brand != 'all':
        referer = f"{base_url}/laptop-{brand}/cat-{cat_id}?pi={page_index}.htm"
    else:
        referer = f"{base_url}/laptop/cat-{cat_id}?pi={page_index}.htm"
    
    headers = {
        "accept": headers_config.get('accept_json', "application/json, text/plain, */*"),
        "accept-language": headers_config.get('accept_language', "vi,en-US;q=0.9,en;q=0.8"),
        "content-type": "application/json",
        "origin": base_url,
        "referer": referer,
        "user-agent": headers_config.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    }
    
    num_row = payload_config.get('num_row', 40)
    start_offset = (page_index - 1) * num_row

    payload = {
        "startOffset": start_offset,
        "numRow": num_row,
        "defaultRow": num_row,
        "categoryIds": [cat_id],
        "merchantIds": [],
        "regionIds": [],
        "isGetResult": True,
        "numPromotedCustomerProduct": 0,
        "propertyFilters": [],
        "propertyRangeFilters": [],
        "keyword": "",
        "productType": "0",
        "isAppend": True,
        "pageIndex": page_index,
        "isDesktop": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for page {page_index}: {e}")
        return None

def find_detail_urls(json_data):
    # A recursive function to search for 'detailUrl' in nested JSON structures
    extracted_urls = []
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == "detailUrl" and isinstance(value, str):
                # Filter out merchant direct links as they don't contain specification tables
                if "direct.htm" not in value:
                    extracted_urls.append(value)
            else:
                extracted_urls.extend(find_detail_urls(value))
    elif isinstance(json_data, list):
        for item in json_data:
            extracted_urls.extend(find_detail_urls(item))
            
    return extracted_urls

def fetch_laptop_html(url, output_filename=None, config=None):
    if config is None:
        config = {}
        
    api_config = config.get('api', {})
    headers_config = api_config.get('headers', {})
    timeout = api_config.get('timeout_seconds', 15)

    headers = {
        "accept": headers_config.get('accept_html', "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"),
        "accept-language": headers_config.get('accept_language', "vi,en-US;q=0.9,en;q=0.8"),
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "upgrade-insecure-requests": "1",
        "user-agent": headers_config.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        html_content = response.text
        
        if output_filename:
            with open(output_filename, "w", encoding="utf-8") as file:
                file.write(html_content)
                
        return html_content
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch HTML from {url}. Error: {e}")
        return None

def extract_table_specifications(html_content):
    if not html_content:
        return {}
        
    # Parse the HTML snippet
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Locate the specific specification table
    table = soup.find('table', class_='table-specifications')
    
    if not table:
        print("Specification table not found in the provided HTML.")
        return {}
        
    extracted_specs = {}
    
    # Iterate through all rows in the table body
    for row in table.find_all('tr'):
        header_cell = row.find('th')
        data_cell = row.find('td')
        
        # Ensure both key and value cells exist before extracting
        if header_cell and data_cell:
            # Strip trailing whitespaces which are common in raw HTML text
            key = header_cell.text.strip()
            value = data_cell.text.strip()
            
            # Handling duplicate keys to prevent overwriting
            original_key = key
            counter = 2
            while key in extracted_specs:
                key = f"{original_key}_{counter}"
                counter += 1
                
            extracted_specs[key] = value
            
    return extracted_specs

def fetch_prices_and_shops(product_id, config=None):
    if config is None: config = {}
    api_config = config.get('api', {})
    headers_config = api_config.get('headers', {})
    timeout = api_config.get('timeout_seconds', 15)
    
    endpoint = api_config.get('compare_endpoint', "https://websosanh.vn/compare-api/get-compare-normal-merchant")
    
    url = f"{endpoint}?rootProductId={product_id}&regionId=0&pageIndex=1&sortType=1&sortRange=0&pageSize=10"
    
    headers = {
        "accept": headers_config.get('accept_json', "application/json, text/plain, */*"),
        "accept-language": headers_config.get('accept_language', "vi,en-US;q=0.9,en;q=0.8"),
        "user-agent": headers_config.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching prices for product {product_id}: {e}")
        return {}
        
    prices_info = {}
    count = 1
    
    companies = data.get('compareVipCompanies', []) + data.get('compareNormalCompanies', [])
    for company in companies:
        if count > 3:
            break
            
        merchant_name = company.get('merchantName', '')
        products = company.get('products', [])
        
        if products:
            # Taking the first product from the merchant
            product = products[0]
            price = product.get('price', '')
            shop_link = product.get('detailUrlMerchant', '')
            
            prices_info[f'shop_{count}_name'] = merchant_name
            prices_info[f'shop_{count}_price'] = price
            prices_info[f'shop_{count}_link'] = shop_link
            count += 1
            
    return prices_info
