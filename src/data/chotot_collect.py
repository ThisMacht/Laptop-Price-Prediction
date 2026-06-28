import requests
from bs4 import BeautifulSoup
import time
import json

def get_product_links(page, config=None):
    """Lấy danh sách link sản phẩm từ trang danh mục."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    }
    url = f"https://www.chotot.com/mua-ban-laptop?page={page}"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tìm các thẻ <a> có class cwv3xk0 theo yêu cầu
        links = []
        a_tags = soup.find_all('a', class_='cwv3xk0')
        for a in a_tags:
            href = a.get('href')
            if href and '.htm' in href:
                # Loại bỏ các tham số theo dõi sau dấu # hoặc ? để có URL sạch
                clean_url = href.split('#')[0].split('?')[0]
                if not clean_url.startswith('http'):
                    clean_url = "https://www.chotot.com" + clean_url
                links.append(clean_url)
        
        return list(set(links)) # Loại bỏ link trùng nếu có
    except Exception as e:
        print(f"Lỗi khi lấy link từ trang {page}: {e}")
        return []

def get_product_details(url, config=None):
    """Truy cập vào link sản phẩm cụ thể để lấy thông tin chi tiết."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        details = {'url': url}
        
        # 1. Lấy giá từ thẻ <b class="p1mdjmwc">
        price_tag = soup.find('b', class_='p1mdjmwc')
        if price_tag:
            details['price'] = price_tag.get_text(strip=True)
            
        # 2. Lấy tiêu đề (thường là h1)
        title_tag = soup.find('h1')
        if title_tag:
            details['title'] = title_tag.get_text(strip=True)

        # 3. Lấy thông tin chi tiết từ container <div class="c189eou8">
        # Các cặp nhãn-giá trị nằm trong <div class="p74axq8">
        spec_items = soup.find_all('div', class_='p74axq8')
        for item in spec_items:
            label_tag = item.find('div', class_='psxqsiz')
            value_tag = item.find('div', class_='p1vpox21')
            
            if label_tag and value_tag:
                label = label_tag.get_text(strip=True).replace(':', '')
                value = value_tag.get_text(strip=True)
                details[label] = value
                
        # Nếu không lấy được bằng class (do web đổi giao diện), thử fallback qua JSON __NEXT_DATA__
        if len(details) <= 2: # Chỉ có url và title/price
            script_tag = soup.find('script', id='__NEXT_DATA__')
            if script_tag:
                try:
                    data = json.loads(script_tag.string)
                    ad_data = data.get('props', {}).get('pageProps', {}).get('adData', {})
                    if ad_data:
                        details['price'] = details.get('price') or ad_data.get('price')
                        params = ad_data.get('parameters', [])
                        for p in params:
                            details[p.get('label')] = p.get('value')
                except:
                    pass
                    
        return details
    except Exception as e:
        print(f"Lỗi khi lấy chi tiết từ {url}: {e}")
        return None
