import os
import tempfile
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

CPPP_CENTRAL_URL = "https://eprocure.gov.in/cppp/latestactivetendersnew/cpppdata"
CPPP_STATE_URL = "https://eprocure.gov.in/cppp/latestactivetendersnew/mmpdata"

import base64

def clean_title_and_ref(raw_text: str, url: str) -> tuple[str, str]:
    # Extract robust ID from URL fragment
    ref = ""
    if url and 'A13h1' in url:
        try:
            parts = url.split('A13h1')
            last = parts[-1]
            last += '=' * (-len(last) % 4)
            ref = base64.b64decode(last).decode('utf-8')
        except Exception:
            pass
            
    if not ref:
        parts = raw_text.split('/')
        if len(parts) > 1:
            ref = parts[-1].strip()
            
    if not ref:
        parts = raw_text.split('[')
        if len(parts) > 1:
            ref = parts[1].replace(']', '').strip()
            
    if not ref:
        ref = "REF-" + str(hash(raw_text))[-8:]

    # For title, let's just use the raw text if it's already clean, or try to clean it
    title = raw_text
    parts = raw_text.split('/')
    if len(parts) > 1 and parts[-1].strip() == ref:
        title = '/'.join(parts[:-1]).strip()
    else:
        parts = raw_text.split('[')
        if len(parts) > 1 and parts[1].replace(']', '').strip() == ref:
            title = parts[0].strip()

    return title, ref

def scrape_tenders(base_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
    """Scrapes active tenders using Selenium and BeautifulSoup with pagination."""
    print(f"Initializing headless Chrome for {base_url}...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    tenders = []
    
    source_name = "cppp" if "cpppdata" in base_url else "mmp"

    try:
        for page in range(1, max_pages + 1):
            url = f"{base_url}?page={page}"
            print(f"Navigating to {url}...")
            driver.get(url)

            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "table"))
                )
            except Exception as e:
                print(f"Timeout waiting for table on page {page}: {e}")
                break

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", id="table")
            if not table:
                print(f"Could not find tender table on page {page}.")
                break

            rows = table.find_all("tr")
            print(f"Found {len(rows)} tender rows on page {page}.")
            
            if len(rows) == 0:
                break

            import dateparser
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                
                try:
                    closing_str = cols[2].text.strip()
                    title_ref = cols[4]
                    title_raw = title_ref.find("a").text.strip() if title_ref.find("a") else title_ref.text.strip()
                    
                    link = title_ref.find("a")["href"] if title_ref.find("a") else ""
                    if link and not link.startswith("http"):
                        link = urllib.parse.urljoin("https://eprocure.gov.in", link)
                    
                    org = cols[5].text.strip() if len(cols) > 5 else "Unknown"
                    title_clean, ref_no = clean_title_and_ref(title_raw, link)

                    closing_dt = dateparser.parse(closing_str)
                    if not closing_dt:
                        continue
                    
                    if closing_dt.tzinfo is None:
                        closing_dt = closing_dt.replace(tzinfo=timezone.utc)

                    tenders.append({
                        "source": source_name,
                        "source_id": ref_no,
                        "title": title_clean,
                        "organization": org,
                        "tender_reference": ref_no,
                        "response_deadline": closing_dt.isoformat(),
                        "source_url": link,
                        "document_urls": [], 
                        "status": "active"
                    })
                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue
    finally:
        driver.quit()
        
    return tenders


def sync_market_tenders(supabase_client):
    """Fetches tenders from Central and State portals and upserts to DB."""
    print("Starting market tender sync...")
    
    central_tenders = scrape_tenders(CPPP_CENTRAL_URL, max_pages=5)
    state_tenders = scrape_tenders(CPPP_STATE_URL, max_pages=5)
    
    all_tenders = central_tenders + state_tenders
    
    upserted_count = 0
    for t in all_tenders:
        try:
            supabase_client.table("market_tenders").upsert(
                t, on_conflict="source,source_id"
            ).execute()
            upserted_count += 1
        except Exception as e:
            print(f"Error upserting tender {t['source_id']}: {e}")

    print(f"Upserted {upserted_count} active tenders.")
    
    now = datetime.now(timezone.utc).isoformat()
    try:
        supabase_client.table("market_tenders").update({"status": "expired"}).lt(
            "response_deadline", now
        ).eq("status", "active").execute()
        print("Marked past-deadline tenders as expired.")
    except Exception as e:
        print(f"Error marking expired tenders: {e}")


async def download_tender_pdf(url: str) -> str:
    """Downloads a PDF from a given URL to a temporary file and returns the path."""
    if not url:
        raise ValueError("No URL provided for PDF download.")
        
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            # Create a named temporary file that is not automatically deleted on close (since we need to pass the path)
            fd, path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)
            return path
    except Exception as e:
        raise RuntimeError(f"Failed to download PDF from {url}: {e}")


def search_market_tenders(supabase_client, query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    """Searches active market tenders. Basic text search for now."""
    db_query = supabase_client.table("market_tenders").select("*").eq("status", "active")
    
    if query:
        # PostgreSQL full text search could be used here via RPC or ilike for simplicity
        db_query = db_query.ilike("title", f"%{query}%")
        
    result = db_query.order("response_deadline", desc=False).limit(limit).execute()
    return result.data
