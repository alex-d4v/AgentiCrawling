import re
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Method 1: Using Requests (Note: this is prone to being blocked by Google)
import urllib.parse
from urllib.parse import urljoin, urlparse

def visit_website(url, timeout=10):
    """
    Visit a website using requests and return the content.
    
    Parameters:
    - url (str): The URL of the website to visit
    - timeout (int): Timeout in seconds for the request
    
    Returns:
    - dict: A dictionary containing status code, text content, and parsed HTML
    """
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    ]
    for ua in user_agents:
        # just take the first result
        # Set a user agent to mimic a browser
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        try:
            # Make the request
            response = requests.get(url, headers=headers, timeout=timeout)
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Parse the HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            return {
                'status_code': response.status_code,
                'content': response.text,
                'soup': soup,
                'headers': dict(response.headers)
            }
        
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }

def duckduckgo_search_requests(query_list):
    results = []
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    ]
    found = 0
    
    for query in query_list:
        for ua in user_agents:
            time.sleep(2)
            parameters = {'q': query.replace(' ', '+')}
            url = 'https://duckduckgo.com/html/'  # DuckDuckGo HTML search endpoint
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.5',
                'User-Agent': ua,
            }
            
            try:
                response = requests.get(url, headers=headers, params=parameters)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract search results - DuckDuckGo HTML structure
                    search_results = []
                    for result in soup.find_all('div', class_='result__body'):
                        link = result.find('a', class_='result__a')
                        if link:
                            href = link.get('href')
                            title = link.text
                            
                            # Extract the actual URL from DuckDuckGo's redirect URL
                            if href and 'uddg=' in href:
                                # Extract the URL after uddg= parameter
                                actual_url = href.split('uddg=')[1].split('&')[0]
                                # URL decode to get the original URL
                                actual_url = urllib.parse.unquote(actual_url)
                                
                                search_results.append({
                                    'title': title,
                                    'url': actual_url
                                })
                    #endfor
                    found += 1
                    results.extend(search_results)
                    print(f"Found {len(search_results)} results for query: {query}")
                    break
                else:
                    print(f"Failed to retrieve results for query: {query}")
                
                # Be respectful with rate limiting
                time.sleep(2)
                
                if found > 3:
                    break
            
            except Exception as e:
                print(f"Error while searching for {query}: {str(e)}")
    
    return results

# Method 2: Using Selenium (more reliable but slower)
def google_search_selenium(query_list):
    results = []
    
    # Set up Chrome options
    chrome_options = Options()
    # Add options as needed, e.g., headless mode
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        for query in query_list:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            
            # Wait for search results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            # Extract search results
            search_results = []
            result_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")
            
            for element in result_elements:
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, "h3")
                    link_element = element.find_element(By.CSS_SELECTOR, "a")
                    
                    title = title_element.text
                    url = link_element.get_attribute("href")
                    
                    if title and url:
                        search_results.append({
                            'query': query,
                            'title': title,
                            'url': url
                        })
                except Exception:
                    # Some elements might not have the expected structure
                    continue
            
            results.extend(search_results)
            print(f"Found {len(search_results)} results for query: {query}")
            
            # Be respectful with rate limiting
            time.sleep(2)
            
    except Exception as e:
        print(f"Error during selenium search: {str(e)}")
    
    finally:
        # Always close the driver
        driver.quit()
    
    return results

def get_all_paragraphs(soup_or_response):
    """
    Extract all paragraphs from a BeautifulSoup object or website response.
    
    Parameters:
    - soup_or_response: BeautifulSoup object or response from visit_website function
    
    Returns:
    - list: List of dictionaries containing paragraph text and their indices
    """
    # Handle input that could be soup object or response dict
    soup = soup_or_response if isinstance(soup_or_response, BeautifulSoup) else soup_or_response.get('soup')
    
    if not soup:
        return []
    
    paragraphs = []
    # Find all paragraph tags
    p_tags = soup.find_all('p')
    
    for idx, p in enumerate(p_tags):
        # Get text and strip whitespace
        text = p.get_text(strip=True)
        if text:  # Only include non-empty paragraphs
            paragraphs.append({
                'index': idx,
                'text': text
            })
    
    return paragraphs

def get_all_urls(soup_or_response, base_url=None):
    """
    Extract all URLs from a BeautifulSoup object or website response.
    
    Parameters:
    - soup_or_response: BeautifulSoup object or response from visit_website function
    - base_url: Base URL to resolve relative URLs (optional)
    
    Returns:
    - list: List of dictionaries containing URLs and their text
    """
    # Handle input that could be soup object or response dict
    if isinstance(soup_or_response, BeautifulSoup):
        soup = soup_or_response
    else:
        soup = soup_or_response.get('soup')
        # Use the response URL as base_url if not provided
        if base_url is None and 'url' in soup_or_response:
            base_url = soup_or_response['url']
    
    if not soup:
        return []
    
    urls = []
    # Find all anchor tags
    a_tags = soup.find_all('a', href=True)
    
    for idx, a in enumerate(a_tags):
        href = a['href']
        text = a.get_text(strip=True)
        
        # Resolve relative URLs if base_url is provided
        if base_url:
            href = urljoin(base_url, href)
        
        # Filter out javascript links and anchors
        if not href.startswith('javascript:') and not href.startswith('#'):
            urls.append({
                'index': idx,
                'url': href,
                'text': text if text else None,
                'parent_element': a.parent.name
            })
    
    return urls

def find_url_with_context(soup_or_response, context_size=1):
    """
    Find URLs matching a pattern and return them with their surrounding text context.
    
    Parameters:
    - soup_or_response: BeautifulSoup object or response from visit_website function
    - url_pattern: String or regex pattern to match in URLs
    - context_size: Number of sibling paragraphs to include before and after
    
    Returns:
    - list: List of dictionaries containing matched URLs and their context
    """
    # Handle input that could be soup object or response dict
    if isinstance(soup_or_response, BeautifulSoup):
        soup = soup_or_response
        base_url = None
    else:
        soup = soup_or_response.get('soup')
        base_url = soup_or_response.get('url')
    
    if not soup:
        return []
    
    # Get all URLs
    all_urls = get_all_urls(soup, base_url)
    results = []
    for url_info in all_urls:
        a_tag = soup.find_all('a', href=True)[url_info['index']]
        # Get the parent container of the link
        parent = a_tag.parent
        # Find the closest container that might have paragraphs
        container = parent
        for _ in range(2):  # Look up to 2 levels up
            if container.find_all('p'):
                break
            if container.parent:
                container = container.parent
            else:
                break
        # Find paragraphs in the container
        paragraphs = container.find_all('p')
        # Find the position the link relative to paragraphs
        link_index = -1
        for i, p in enumerate(paragraphs):
            if p.find(a_tag):
                link_index = i
                break
        # If link is inside a paragraph, use that as context
        context_paragraphs=None
        if link_index >= 0:
            start_idx = max(0, link_index - context_size)
            end_idx = min(len(paragraphs), link_index + context_size + 1)
            context_paragraphs = paragraphs[start_idx:end_idx]
        #else:
            # If link is not in paragraph, get closest paragraphs
            #closest_p = min(paragraphs, key=lambda p: abs(p.sourceline - a_tag.sourceline)) if paragraphs else None
            #context_paragraphs = [closest_p] if closest_p else []
        # Extract text from context paragraphs
        context_text=None
        if context_paragraphs:
            context_text = [p.get_text(strip=True) for p in context_paragraphs]
        results.append({
            'url': url_info['url'],
            'link_text': url_info['text'],
            'context': context_text if context_text else '',
        })
    
    return results