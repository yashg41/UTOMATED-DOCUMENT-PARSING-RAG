#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document Scraper Module

This module opens a website, navigates to specific pages, extracts table data,
and downloads files to designated folders.
"""

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, NoSuchFrameException,
    StaleElementReferenceException, ElementClickInterceptedException
)
import traceback
from datetime import datetime
import time
import re
import os
import sys
import requests
import json
import zipfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration for Web Scraper ---
# Wait time for elements
DEFAULT_WAIT_TIME = 30  # Generous wait time

# URLs and Credentials
BASE_URL = "https://docmap.synergymarinegroup.com/Docmap80/page/doc/dmDocIndex.html"
START_PAGE_URL = "https://docmap.synergymarinegroup.com/Docmap80/page/doc/dmfDocAllContainer.html?CONTENT=moreNews.html?SWITCH=1?C_PAGE=moreNews.html"
# Note: Using lowercase as in the working notebook code
new_documents_page_url = "https://docmap.synergymarinegroup.com/Docmap80/page/doc/moreNews.html?CONTENT=moreNews.html"
# Changed to templates for dynamic construction
CONTENT_PAGE_URL_TEMPLATE = "https://docmap.synergymarinegroup.com/Docmap80/page/doc/dmDocIndex.html?DOC%2FISSUE+DATE%3B3=24%2F04%2F2025?SEARCHRESULT=YES?CONTENT=dmDocContent.html?DOCSTATUS=4872"
FINAL_NAV_URL_TEMPLATE = "https://docmap.synergymarinegroup.com/Docmap80/page/doc/dmDocContentContent1.html?SPN=1?DOC%2FISSUE+DATE%3B3=24%2F04%2F2025?SEARCHRESULT=YES?CONTENT=dmDocContent.html?DOCSTATUS=4872?module=DOC"

# API Configuration
API_BASE_URL = "https://docmap.synergymarinegroup.com/Docmap80/rest/"
AUTH_ENDPOINT = 'authentication.json'
DOWNLOAD_ENDPOINT = 'docs/downloaddocs.zip'
DOM_SELECTOR = '0xF8'  # Example selector, adjust if needed

def extract_date(text):
    """
    Extracts date in dd/mm/yyyy format from a string.
    Returns the date string or None if not found or input is not a string.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
    if match:
        return match.group(1)
    return None

def login_extract_process(base_url, start_page_url, new_documents_page_url, content_url_template, final_nav_url_template, username, password):
    """
    Logs in, navigates, clicks, extracts table data, and processes it.
    Returns processed DataFrame on success, None on failure.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Uncomment for headless execution
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")  # Optional: for debugging
    options.add_argument("--disable-software-rasterizer")  # May help in some environments
    options.add_argument("--start-maximized")  # Ensure browser window is maximized

    driver = None  # Initialize driver for finally block
    print("Launching browser...")
    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, DEFAULT_WAIT_TIME)
        print("Browser launched.")

        # === Login ===
        print(f"\n[Step 1] Navigating to login page: {base_url}")
        driver.get(base_url)
        print(f"Initial URL: {driver.current_url}")
        wait.until(EC.presence_of_element_located((By.NAME, 'USER')))
        print("Login page loaded.")

        print("\n[Step 2] Entering login credentials...")
        user_field = wait.until(EC.visibility_of_element_located((By.NAME, 'USER')))
        user_field.send_keys(username)
        pass_field = wait.until(EC.visibility_of_element_located((By.NAME, '_PASSWORD')))
        pass_field.send_keys(password)
        signin_button = wait.until(EC.element_to_be_clickable((By.ID, 'Sign in_fsubmit();')))
        signin_button.click()
        print("Login submitted.")
        print("Waiting for main frame structure page to load...")
        wait.until(EC.url_contains('dmDocIndex.html'))  # Wait for redirection after login
        print(f"URL after login: {driver.current_url}")

        # === Navigation ===
        print(f"\n[Step 3] Navigating to start page URL: {start_page_url}")
        driver.get(start_page_url)
        print("Waiting for start page URL to load...")
        print(f"✅ Browser navigated to start page URL: {driver.current_url}")

        print(f"\n[Step 4] Navigating to new documents page: {new_documents_page_url}")
        driver.get(new_documents_page_url)
        print("Waiting for new documents page to load...")
        print(f"✅ Browser navigated to new documents page: {driver.current_url}")

        print(f"\n[Step 5] Searching for 'Last 7 days' button...")
        button_xpath = "//div[normalize-space(.)='Last 7 days']"
        
        # Track window handles for proper new tab handling
        original_window = driver.current_window_handle
        print(f"Original window handle: {original_window}")
        num_windows_before_click = len(driver.window_handles)
        print(f"Number of windows before click: {num_windows_before_click}")

        last_7_days_button = wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
        print("Found 'Last 7 days' element (div) and it is clickable.")
        last_7_days_button.click()
        print("✅ Clicked 'Last 7 days' element.")
        
        print("Waiting for new window to open...")
        # Wait for new window/tab to open
        wait.until(EC.number_of_windows_to_be(num_windows_before_click + 1))
        
        all_windows = driver.window_handles
        print(f"All window handles after click: {all_windows}")
        print(f"Number of windows after click: {len(all_windows)}")
        new_tab_url = None

        # Find and switch to the new tab/window
        if len(all_windows) > num_windows_before_click:
            new_window = None
            for window in all_windows:
                if window != original_window:
                    new_window = window
                    break
            
            if new_window:
                print(f"Switching to new window: {new_window}")
                driver.switch_to.window(new_window)
                print(f"Switched to new tab/window.")
                time.sleep(2)  # Allow new page to stabilize URL
                new_tab_url = driver.current_url
                print(f"✅ URL of the new tab: {new_tab_url}")
            else:
                print("Could not identify the new window even though window count increased.")
        else:
            print("No new tab/window was opened or detected.")
        
        print("Pausing briefly after Last 7 days click operations...")
        time.sleep(3)  # Pause for page transition
        print(f"Current URL after operations: {driver.current_url}")

        # Verify we have the new URL to proceed
        if not new_tab_url:
            error_msg = "❌ FAILED: 'new_tab_url' was not captured after clicking 'Last 7 days'. Cannot proceed with dynamic URL steps."
            print(error_msg)
            raise Exception(error_msg)

        print(f"\n[Step 6] Navigating to content URL (from new tab): {new_tab_url}")
        driver.get(new_tab_url)  # Use the dynamically captured URL
        print("Waiting for content URL page to load...")
        wait.until(EC.url_contains("dmDocIndex.html"))
        print(f"✅ Browser navigated to content URL: {driver.current_url}")

        # Extract the dynamic date parameter for constructing the final URL
        print("\n[Step 7] Constructing and navigating to final target URL...")
        date_param_match = re.search(r"(DOC%2FISSUE\+DATE%3B3=\d{2}%2F\d{2}%2F\d{4})", new_tab_url)
        if not date_param_match:
            error_msg = f"❌ FAILED: Could not extract dynamic date parameter from URL: {new_tab_url}"
            print(error_msg)
            raise Exception(error_msg)
        
        dynamic_date_query_param = date_param_match.group(1)
        print(f"Extracted dynamic date parameter: {dynamic_date_query_param}")

        # Construct the final navigation URL with the extracted date parameter
        final_nav_url = re.sub(
            r"DOC%2FISSUE\+DATE%3B3=\d{2}%2F\d{2}%2F\d{4}",
            dynamic_date_query_param,
            final_nav_url_template
        )
        
        print(f"Constructed final navigation URL: {final_nav_url}")
        driver.get(final_nav_url)
        print("Waiting for final URL page to load/stabilize...")
        wait.until(EC.presence_of_element_located((By.ID, 'searchResultTable')))
        print(f"✅ Browser navigated to final URL: {driver.current_url}")
        time.sleep(2)  # Short pause for rendering

        # === Extract Table Data ===
        print("\n[Step 8] Extracting data from table 'searchResultTable'...")
        print("Parsing page source...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', id='searchResultTable')
        if not table:
            raise Exception("Table with id 'searchResultTable' not found in parsed HTML.")

        print("Extracting table data...")
        data = []
        doc_ids = []
        rows = table.find_all('tr')
        for row in rows:
            doc_id = row.get('data-docid')  # Get DocID from attribute
            doc_ids.append(doc_id)
            cols = row.find_all(['th', 'td'])
            cols = [ele.text.strip() for ele in cols]
            data.append(cols)

        if len(data) < 2:
            raise Exception("Table has no data rows (only header or empty).")

        # Create initial DataFrame
        df_raw = pd.DataFrame(data[1:], columns=data[0])  # Use header row for columns
        df_raw['DocID'] = doc_ids[1:]  # Add DocID column from extracted attributes
        print(f"✅ Successfully extracted raw table data ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns).")

        # === Process Extracted DataFrame ===
        print("\n[Step 9] Processing extracted DataFrame...")
        df_processed = df_raw.copy()  # Work on a copy

        # Drop first column by index (often an empty selection column)
        if not df_processed.empty and df_processed.shape[1] > 0:
            print(f"Dropping first column by index: '{df_processed.columns[0]}'")
            df_processed = df_processed.drop(df_processed.columns[0], axis=1)
        else:
            print("Warning: DataFrame is empty or has no columns, skipping first column drop.")

        # Drop specific columns by name if they exist
        cols_to_drop_by_name = ['Issued, not yet valid', 'Replaced by a newer version', 'Withdrawn']
        existing_cols_to_drop = [col for col in cols_to_drop_by_name if col in df_processed.columns]
        if existing_cols_to_drop:
            print(f"Dropping columns by name: {existing_cols_to_drop}")
            df_processed = df_processed.drop(existing_cols_to_drop, axis=1)
        else:
            print(f"Columns {cols_to_drop_by_name} not found, skipping drop by name.")

        # Filter rows where DocID is not null/None
        if "DocID" in df_processed.columns:
            initial_rows = df_processed.shape[0]
            print("Filtering rows where DocID is not null...")
            df_processed = df_processed[df_processed["DocID"].notna() & (df_processed["DocID"] != '')]
            print(f"Removed {initial_rows - df_processed.shape[0]} rows with null or empty DocID.")
        else:
            print("Warning: 'DocID' column not found, skipping null filter.")

        # Rename columns positionally
        print(f"Current columns before potential rename: {df_processed.columns.tolist()}")
        expected_col_count_before_rename = 8  # As per original script logic
        if df_processed.shape[1] == expected_col_count_before_rename:
            print(f"Renaming {df_processed.shape[1]} columns positionally...")
            new_column_names = ["Doc. No", "Document title", "Type", "Validity area", "Date", "w", "a", "DocID"]
            df_processed.columns = new_column_names
            print(f"Columns renamed to: {df_processed.columns.tolist()}")

            # Drop the newly named 'w' and 'a' columns
            cols_to_drop_w_a = ['w', 'a']
            print(f"Dropping columns: {cols_to_drop_w_a}")
            existing_w_a = [col for col in cols_to_drop_w_a if col in df_processed.columns]
            if existing_w_a:
                df_processed = df_processed.drop(existing_w_a, axis=1)
                print(f"Columns after dropping 'w', 'a': {df_processed.columns.tolist()}")
            else:
                print("Columns 'w', 'a' not found after rename, skipping drop.")
        else:
            print(f"⚠️ WARNING: Expected {expected_col_count_before_rename} columns for positional rename, but found {df_processed.shape[1]}. Skipping positional rename.")
            print(f"Current columns remain: {df_processed.columns.tolist()}")

        # Date Extraction Processing
        date_column_name = 'Date'  # This name depends on successful renaming above
        if date_column_name in df_processed.columns:
            print(f"\nExtracting dates from '{date_column_name}' column...")
            df_processed[date_column_name] = df_processed[date_column_name].apply(extract_date)
            print("✅ Date extraction applied.")
        else:
            print(f"\n⚠️ Warning: Column '{date_column_name}' not found. Skipping date extraction.")
            print(f"Available columns for date extraction check: {df_processed.columns.tolist()}")

        print("✅ Data processing finished.")
        
        # Save the processed DataFrame
        csv_output_path = "docmap_final_processed.csv"
        df_processed.to_csv(csv_output_path, index=False)
        print(f"✅ Processed data saved to {csv_output_path}")
        
        return df_processed

    except TimeoutException as te:
        print(f"❌ FAILED: A timeout occurred - {te}")
        print(f"Last known URL: {driver.current_url if driver else 'N/A'}")
        traceback.print_exc()
    except (NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException) as se:
        print(f"❌ FAILED: Could not find or interact with an element - {se}")
        print(f"Last known URL: {driver.current_url if driver else 'N/A'}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        print(f"Last known URL: {driver.current_url if driver else 'N/A'}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---")
        try:
            if driver:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                screenshot_file = f'error_screenshot_{timestamp}.png'
                driver.save_screenshot(screenshot_file)
                print(f"Saved error screenshot: {screenshot_file}")
        except Exception as ss_err:
            print(f"Could not save error screenshot: {ss_err}")
        return None

    finally:
        print("\n[Final Step] Scraping finished.")
        if driver:
            try:
                driver.quit()
                print("Browser closed.")
            except Exception as quit_err:
                print(f"⚠️ Error occurred trying to quit browser: {quit_err}")
        else:
            print("Driver was not successfully initialized.")

class DocumentScraper:
    """Main document scraper class that handles website interaction and file downloads."""
    
    def __init__(self, username, password, domain="docMap", output_dir="downloaded_data"):
        """Initialize with credentials and output directory."""
        self.username = username
        self.password = password
        self.domain = domain
        self.output_dir = output_dir
        self.downloaded_zip_path = os.path.join(os.getcwd(), f"{output_dir}.zip")
        self.extraction_directory = output_dir
        self.df_final = None
        self.session_id = None
        
        # Create output directory
        os.makedirs(self.extraction_directory, exist_ok=True)
        print(f"Files will be extracted to: {self.extraction_directory}")
    
    def login_and_scrape_data(self):
        """Logs into the website, navigates to the required pages, and extracts data."""
        # Using the standalone function from the notebook that we know works
        df_processed = login_extract_process(
            BASE_URL, 
            START_PAGE_URL, 
            new_documents_page_url, 
            CONTENT_PAGE_URL_TEMPLATE,  # Updated to template 
            FINAL_NAV_URL_TEMPLATE,     # Updated to template
            self.username, 
            self.password
        )
        
        # Store the final DataFrame for later use
        self.df_final = df_processed
        return df_processed
    
    def authenticate_api(self):
        """Authenticates with the API and obtains a session token."""
        print("\n--- [Step 1] Authenticating with API ---")
        # Use download credentials instead of scraper credentials
        downloader_username = os.getenv("DOWNLOADER_USERNAME")
        downloader_password = os.getenv("DOWNLOADER_PASSWORD")
        
        # If downloader credentials aren't set, fall back to scraper credentials
        if not downloader_username:
            print("DOWNLOADER_USERNAME not found in environment, using scraper username")
            downloader_username = self.username
            
        if not downloader_password:
            print("DOWNLOADER_PASSWORD not found in environment, using scraper password")
            downloader_password = self.password
            
        auth_payload = {
            'user': downloader_username,
            'password': downloader_password,
            'domain': self.domain
        }
        
        # Mask credentials in logs
        masked_username = downloader_username if downloader_username else "None"
        masked_password = "********" if downloader_password else "None"
        print(f"Authenticating with username: {masked_username}")
        
        url = f"{API_BASE_URL}{AUTH_ENDPOINT}"
        if DOM_SELECTOR:
            url += f"?domselector={DOM_SELECTOR}"

        print(f"Sending POST request to: {url}")
        try:
            response = requests.post(url, json=auth_payload)
            print(f"Response Status Code: {response.status_code}")
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Process response
            auth_response = response.json()
            if (isinstance(auth_response, dict) and 'result' in auth_response 
                and isinstance(auth_response['result'], dict) 
                and 'sessionId' in auth_response['result']):
                
                session_info = auth_response['result']
                self.session_id = session_info['sessionId']
                print(f"✅ Authentication successful. Session ID obtained: {self.session_id}")
                return True
            else:
                print(f"❌ Authentication failed or session ID not found in response: {auth_response}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR during request to {url}: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error during authentication: {e}")
            traceback.print_exc()
            return False
            
    def download_documents(self):
        """Downloads documents based on the DocIDs in the scraped DataFrame."""
        # Verify we have data to download
        if self.df_final is None or self.df_final.empty:
            print("❌ No data available for download. Please run login_and_scrape_data first.")
            return False
            
        # Authenticate first if needed
        if self.session_id is None:
            if not self.authenticate_api():
                return False
                
        # Prepare DocIDs for download
        print("\n--- [Step 2] Preparing Download Request ---")
        list_id = self.df_final['DocID'].astype(str).tolist()
        formatted_ids_for_api = json.dumps([{"id": id_} for id_ in list_id])
        print(f"Prepared {len(list_id)} DocIDs for download request.")
        
        # Download files
        print("\n--- [Step 3] Downloading Files ---")
        url = f"{API_BASE_URL}{DOWNLOAD_ENDPOINT}"
        headers = {"DMTOKEN": self.session_id}
        payload = {
            "FILES2DOWNLOAD": formatted_ids_for_api,
            "DOCSFORMAT": "original"
        }
        
        print(f"Sending POST request to download files: {url}")
        try:
            response = requests.post(url, headers=headers, data=payload, stream=True)
            print(f"Download Response Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print(f"Attempting to save downloaded file to: {self.downloaded_zip_path}")
                with open(self.downloaded_zip_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                print(f"✅ File download successful.")
                
                # Extract the downloaded zip
                return self.extract_files()
            else:
                print(f"❌ Download failed with status {response.status_code}.")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR during file download request: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error during download: {e}")
            traceback.print_exc()
            return False
    
    def extract_files(self):
        """Extracts downloaded zip file and organizes by document type."""
        print("\n--- [Step 4] Unzipping Downloaded File ---")
        
        if not os.path.exists(self.downloaded_zip_path):
            print(f"❌ Error: Downloaded zip file not found at {self.downloaded_zip_path}")
            return False
            
        if os.path.getsize(self.downloaded_zip_path) == 0:
            print(f"❌ Error: Downloaded zip file is empty (0 bytes)")
            return False
            
        # Create DocID to Type mapping for extracted files
        docid_to_type = {}
        if 'Type' in self.df_final.columns and 'DocID' in self.df_final.columns:
            docid_to_type = dict(zip(self.df_final['DocID'], self.df_final['Type']))
            print(f"Created mapping of {len(docid_to_type)} DocIDs to their respective Types.")
        else:
            print("Warning: 'Type' column not found in data. Files will be extracted to the root directory.")
        
        # First analyze the zip to find which document types actually have files
        document_types_with_files = set()
        try:
            with zipfile.ZipFile(self.downloaded_zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                        
                    filename = os.path.basename(file_info.filename)
                    # Get DocID from filename prefix (assuming format: DOCID_filename)
                    docid = None
                    if '_' in filename:
                        docid_part = filename.split('_')[0]
                        if docid_part.isdigit():
                            docid = docid_part
                    
                    # Add document type to set if there's a file for this type
                    if docid and docid in docid_to_type:
                        doc_type = docid_to_type[docid]
                        document_types_with_files.add(doc_type)
            
            # Only create folders for document types that actually have files
            if document_types_with_files:
                print(f"Found {len(document_types_with_files)} document types with files: {', '.join(document_types_with_files)}")
                for doc_type in document_types_with_files:
                    type_folder = os.path.join(self.extraction_directory, doc_type)
                    os.makedirs(type_folder, exist_ok=True)
                    print(f"Created subfolder for type '{doc_type}': {type_folder}")
            else:
                print("No document types with files found.")
        except zipfile.BadZipFile:
            print(f"❌ ERROR: '{self.downloaded_zip_path}' is not a valid zip file or is corrupted.")
            return False
        except Exception as e:
            print(f"❌ An unexpected error during zip analysis: {e}")
            traceback.print_exc()
            return False
            
        # Extract files
        extracted_files_info = []
        try:
            with zipfile.ZipFile(self.downloaded_zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                        
                    filename = os.path.basename(file_info.filename)
                    # Get DocID from filename prefix (assuming format: DOCID_filename)
                    docid = None
                    if '_' in filename:
                        docid_part = filename.split('_')[0]
                        if docid_part.isdigit():
                            docid = docid_part
                    
                    # Determine target folder based on document type
                    if docid and docid in docid_to_type:
                        doc_type = docid_to_type[docid]
                        target_dir = os.path.join(self.extraction_directory, doc_type)
                    else:
                        target_dir = self.extraction_directory
                        
                    # Extract the file
                    source = zip_ref.open(file_info)
                    target_path = os.path.join(target_dir, filename)
                    
                    # Create parent directory if it doesn't exist (just in case)
                    parent_dir = os.path.dirname(target_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    with open(target_path, "wb") as target:
                        target.write(source.read())
                    
                    print(f"Extracted file {filename} to {doc_type if docid and docid in docid_to_type else 'root'} folder")
                    extracted_files_info.append({
                        'File_Path': target_path,
                        'DocID': docid if docid else 'Unknown',
                        'Type': docid_to_type.get(docid, 'Unknown') if docid else 'Unknown'
                    })
            
            print(f"✅ Successfully extracted {len(extracted_files_info)} file(s) with type-based organization.")
            
            # Create DataFrame of extracted files
            files_df = pd.DataFrame(extracted_files_info)
            print("\nExtracted Files DataFrame (head):")
            print(files_df.head())
            
            # Merge with original data for comprehensive output
            if not files_df.empty:
                # Merge based on DocID
                result_df = pd.merge(
                    self.df_final, 
                    files_df[['File_Path', 'DocID']], 
                    on='DocID', 
                    how='left'
                )
                
                # Extract document names from file paths
                result_df['Document Name'] = result_df['File_Path'].apply(
                    lambda x: os.path.splitext(os.path.basename(str(x)))[0] if pd.notna(x) else None
                )
                
                # Save the final result
                result_df.to_csv("final_output_summary.csv", index=False)
                print("\n✅ Final summary saved to 'final_output_summary.csv'")
                
            return True
            
        except zipfile.BadZipFile:
            print(f"❌ ERROR: '{self.downloaded_zip_path}' is not a valid zip file or is corrupted.")
            return False
        except Exception as e:
            print(f"❌ An unexpected error during extraction: {e}")
            traceback.print_exc()
            return False

def scrape_documents(username, password, domain="docMap", output_dir="downloaded_data"):
    """
    Main function to execute the document scraping and downloading process.
    
    Args:
        username (str): Username for login
        password (str): Password for login
        domain (str): Domain for authentication, default is "docMap"
        output_dir (str): Directory to save downloaded files, default is "downloaded_data"
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = datetime.now()
    print("--- Starting Document Scraping and Download Process ---")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    scraper = DocumentScraper(username, password, domain, output_dir)
    
    # Step 1: Login and scrape data
    df = scraper.login_and_scrape_data()
    if df is None:
        print("❌ Document scraping failed. Cannot proceed with downloads.")
        return False
        
    # Step 2: Download and extract documents
    success = scraper.download_documents()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n--- Process Summary ---")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Status: {'✅ Completed successfully' if success else '❌ Finished with errors'}")
    
    return success

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # First check if env vars are set
    env_username = os.getenv("SCRAPER_USERNAME")
    env_password = os.getenv("SCRAPER_PASSWORD")

    # Debug information about environment variables
    print("---- SCRAPER CREDENTIALS DEBUG INFO ----")
    print(f"SCRAPER_USERNAME environment variable set: {'Yes' if env_username else 'No'}")
    print(f"SCRAPER_PASSWORD environment variable set: {'Yes' if env_password else 'No'}")
    print(f".env file exists in current directory: {'Yes' if os.path.exists('.env') else 'No'}")
    if env_username:
        print(f"Using SCRAPER_USERNAME from environment: {env_username}")
    else:
        print("WARNING: SCRAPER_USERNAME not found in environment variables!")
    
    if not env_username or not env_password:
        print("ERROR: Missing required environment variables. Please set SCRAPER_USERNAME and SCRAPER_PASSWORD in your .env file.")
        sys.exit(1)
    
    print("---------------------------")
    
    # Execute
    scrape_documents(env_username, env_password) 