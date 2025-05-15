import json
import os
import glob
import traceback
import asyncio
import time
import random
import logging
from typing import List, Dict, Optional, Any
import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import nest_asyncio  # Import for nested event loops

# Apply patch to allow nested event loops
nest_asyncio.apply()

# Import error classes directly from openai in v1.3.5
from openai import (
    OpenAIError, 
    RateLimitError, 
    Timeout, 
    APIError
)

# --- Configuration ---
BASE_DIRECTORY = 'parsed_json_results'
SUBFOLDERS = ['ISM Manual', 'OFFICE MANUAL', 'Policies', 'Sample Document Type']
# LLM Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 30
CONCURRENT_REQUESTS = 3  # Number of concurrent requests to OpenAI API
# --- ---

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize AsyncOpenAI client with v1.3.5
async_client = AsyncOpenAI(api_key=openai_api_key)

# --- Removed Summary Generation Prompts ---

def clean_text_for_llm(text: str) -> str:
    """Clean and prepare text for sending to LLM."""
    if not text:
        return ""
    # Remove excessive whitespace and normalize
    cleaned = " ".join(text.split())
    return cleaned

async def extract_json_with_llm_async(
    text_content: str,
    model: str,
    semaphore: asyncio.Semaphore,
    item_info: dict  # Pass file/page info for logging
) -> dict | None:
    """ Sends text to LLM, attempts to parse JSON, with backoff and timeout. """
    if not async_client:
        logging.error("OpenAI client not initialized. Skipping LLM call.")
        return None

    item_log_prefix = f"[File: {item_info.get('file', 'N/A')}, Page: {item_info.get('page', 'N/A')}]"
    cleaned_content = clean_text_for_llm(text_content)
    if not cleaned_content:
        logging.warning(f"{item_log_prefix} Input text was empty after cleaning. Skipping LLM call.")
        return None

    # Focused prompt asking for the flattened structure with all content in a single field
    system_prompt = """You are an expert JSON extractor. Analyze the user's text, which represents data for a single page.
    Your goal is to extract a JSON structure with ALL of the following keys (even if they're not found in the text):
    - documentName (title of the document)
    - revDate (revision date in YYYY-MM-DD format if possible)
    - RevNo (revision number)
    - reviewDate (review date in YYYY-MM-DD format if possible)
    - chapter (chapter or section title)
    - section (subsection name)
    - content (ALL text content including tables and any other information from the page)
    
    IMPORTANT: ALL fields must be present in your output. If you can't find a value for a field, use an empty string ("").
    Do not omit any fields - include them all, even if they are empty strings.
    
    The content field is EXTREMELY IMPORTANT. It must include ALL text content including tables, lists and any structured information from the page. DO NOT split content across multiple fields or structures - put everything in a single content field.
    
    Your response MUST be ONLY the JSON object itself, without any surrounding text or markdown formatting (like ```json ... ```).
    
    Key requirements:
    1. **Valid JSON:** Strictly adhere to JSON syntax.
    2. **String Escaping:** Correctly escape special characters (", \\, \\n, etc.) within strings.
    3. **Flat Structure:** Do NOT use nested objects or arrays; all fields should be at the top level.
    4. **ALL Fields Required:** ALWAYS include every field listed above. Use "" for missing values.
    5. **Full Content Preservation:** Preserve tables in markdown format, include all text from the page."""

    user_prompt = f"Extract the structured JSON data for this page. ALL fields (documentName, revDate, RevNo, reviewDate, chapter, section, content) must be included even if empty. The content field MUST contain ALL text (including tables, lists, etc.) from the page:\n\n{cleaned_content}"

    retries = 0
    backoff_time = INITIAL_BACKOFF_SECONDS

    while retries < MAX_RETRIES:
        async with semaphore:
            logging.info(f"{item_log_prefix} Attempt {retries + 1}/{MAX_RETRIES}: Sending request to {model}...")
            llm_output_raw = None
            request_start_time = time.time()
            try:
                # Standard API call for v1.3.5
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Low temp for deterministic extraction
                )
                
                request_duration = time.time() - request_start_time
                llm_output_raw = response.choices[0].message.content
                logging.info(f"{item_log_prefix} LLM Response Received (Length: {len(llm_output_raw or '')}, Duration: {request_duration:.2f}s)")

                if not llm_output_raw:
                    logging.error(f"{item_log_prefix} LLM returned an empty response.")
                    return None  # Treat as failure for this item

                try:
                    # Attempt to parse the raw string output as JSON
                    extracted_data = json.loads(llm_output_raw)
                    if isinstance(extracted_data, dict):
                        # Ensure 'page' number from original item is included/correct
                        original_page = item_info.get('page')
                        llm_page = extracted_data.get('page')

                        if original_page is not None:
                            if llm_page is None:
                                extracted_data['page'] = original_page
                            elif llm_page != original_page:
                                logging.warning(f"{item_log_prefix} LLM data page ({llm_page}) differs from original ({original_page}). Overwriting with original page number.")
                                extracted_data['page'] = original_page
                        
                        # Rename documentTitle to documentName if present
                        if 'documentTitle' in extracted_data and 'documentName' not in extracted_data:
                            extracted_data['documentName'] = extracted_data.pop('documentTitle')

                        # Ensure all required fields are present
                        required_fields = ['documentName', 'revDate', 'RevNo', 'reviewDate', 'chapter', 'section', 'content']
                        for field in required_fields:
                            if field not in extracted_data:
                                extracted_data[field] = ""
                                logging.warning(f"{item_log_prefix} Added missing field '{field}' with empty string")

                        # Additional check for content field - it should always have something
                        if not extracted_data['content']:
                            # Try to extract tables or anything that looks like content 
                            # from the original text as fallback
                            tables = []
                            for line in cleaned_content.split('\n'):
                                if line.strip().startswith('|'):
                                    tables.append(line)
                            
                            if tables:
                                extracted_data['content'] = '\n'.join(tables)
                            else:
                                # Last resort - just return a subset of the original text
                                extracted_data['content'] = cleaned_content[:500] + "..."
                                
                            logging.warning(f"{item_log_prefix} Added content from original text as fallback")

                        logging.info(f"{item_log_prefix} Successfully parsed LLM response as JSON dict.")
                        return extracted_data
                    else:
                        logging.error(f"{item_log_prefix} LLM response parsed, but not a dictionary (type: {type(extracted_data)}). Raw: {llm_output_raw[:200]}...")
                        return None  # Structure is wrong
                except json.JSONDecodeError as json_err:
                    logging.error(f"{item_log_prefix} Failed to parse LLM response as JSON. Error: {json_err}. Raw snippet: {llm_output_raw[max(0, json_err.pos-20):json_err.pos+20]}")
                    return None  # Parse errors usually indicate bad LLM output

            # --- Error Handling & Retries ---
            except RateLimitError as rle:
                error_type, error_details = "RateLimitError", rle
            except Timeout as te:
                error_type, error_details = "Timeout", te
            except APIError as apie:
                # Retry only on 5xx errors (server-side issues) or 429 (rate limit, handled by RateLimitError too but catch here just in case)
                if (apie.status_code >= 500 or apie.status_code == 429) and retries < MAX_RETRIES - 1:
                    error_type, error_details = f"APIError (Retryable - {apie.status_code})", apie.message
                else:
                    logging.error(f"{item_log_prefix} OpenAI API Error (Non-Retried/Final). Status={apie.status_code}, Message={apie.message}")
                    return None  # Non-retryable or final retry failed
            except Exception as e:
                # Catch potential client-side errors (network issues, config problems)
                logging.error(f"{item_log_prefix} Unexpected error during LLM API call: {e.__class__.__name__} - {e}\n{traceback.format_exc(limit=3)}")
                # Decide if retry makes sense for generic exceptions? Often not.
                return None  # Unexpected errors usually not retryable safely

            # --- If Retryable Error Occurred ---
            retries += 1
            if retries >= MAX_RETRIES:
                logging.error(f"{item_log_prefix} Exceeded max retries ({MAX_RETRIES}) for {error_type}. Giving up. Last error: {error_details}")
                return None
            # Exponential backoff with jitter
            wait_time = min(MAX_BACKOFF_SECONDS, backoff_time * (2 ** (retries - 1))) + random.uniform(0, 1)
            logging.warning(f"{item_log_prefix} {error_type}. Retrying in {wait_time:.2f}s (Attempt {retries + 1}/{MAX_RETRIES})...")
            await asyncio.sleep(wait_time)
            # Loop continues for the next attempt

    # This point should ideally not be reached if the loop logic is correct
    logging.error(f"{item_log_prefix} Failed LLM extraction permanently after {MAX_RETRIES} retries (Loop exited unexpectedly).")
    return None

# --- Removed Summary Generation Functions ---

async def process_file_with_llm(input_filepath, model="gpt-4o", semaphore=None):
    """
    Reads a JSON file containing page and md data, extracts structured information using LLM,
    and saves the result back to the same file.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)  # Default to single concurrency if not provided
        
    print(f"\n--- Processing file with LLM: {os.path.basename(input_filepath)} ---")
    processed_successfully = False

    # Skip job_info.json files
    if os.path.basename(input_filepath) == 'job_info.json':
        print(f"   Skipping job_info.json file")
        return False

    try:
        # Check if the file exists before trying to open it
        if not os.path.exists(input_filepath):
            print(f"   Error: Input file path does not exist: {input_filepath}")
            return False

        # Open and read the input JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Input file loaded: {os.path.basename(input_filepath)}")

        # Check if the data is in the expected format (list of dicts with 'page' and 'md')
        if not isinstance(data, list) or not all(isinstance(item, dict) and 'page' in item and 'md' in item for item in data):
            print(f"   Warning: File {os.path.basename(input_filepath)} is not in the expected format (list of dicts with 'page' and 'md'). Skipping.")
            return False

        # Process each page with the LLM
        enhanced_data = []
        file_basename = os.path.basename(input_filepath)
        
        # Create tasks for LLM processing
        tasks = []
        for item in data:
            page_num = item.get('page')
            md_content = item.get('md')
            
            if not md_content:
                print(f"   Warning: Empty 'md' content for page {page_num} in {file_basename}. Skipping this page.")
                continue
                
            # Check if the md content already contains structured JSON (starts with ```json)
            # If so, try to extract it directly first
            if md_content.strip().startswith('```json'):
                try:
                    # Extract JSON content between triple backticks
                    json_content = md_content.strip().replace('```json', '', 1)
                    json_content = json_content.strip()
                    if json_content.endswith('```'):
                        json_content = json_content[:-3].strip()
                    
                    # Try to parse the JSON content
                    parsed_json = json.loads(json_content)
                    
                    # If successful, create a flattened structure with the important fields
                    if isinstance(parsed_json, dict):
                        flattened_json = {}
                        
                        # Add page number
                        flattened_json['page'] = page_num
                        
                        # Copy top-level fields from parsed_json
                        for key in ['documentTitle', 'revDate', 'RevNo', 'reviewDate', 'chapter']:
                            if key in parsed_json:
                                # Handle documentTitle vs documentName - use documentName in output
                                if key == 'documentTitle':
                                    flattened_json['documentName'] = parsed_json[key]
                                else:
                                    flattened_json[key] = parsed_json[key]
                        
                        # Simply concatenate all section content into a single content field
                        if 'sections' in parsed_json and isinstance(parsed_json['sections'], list):
                            section_contents = []
                            for section in parsed_json['sections']:
                                if 'section_name' in section and section['section_name']:
                                    flattened_json['section'] = section['section_name']  # Use the first non-empty section name
                                    
                                if 'content' in section and section['content']:
                                    section_contents.append(section['content'])
                            
                            # Join all section contents into a single content field
                            if section_contents:
                                flattened_json['content'] = '\n\n'.join(section_contents)
                        
                        # Ensure all required fields are present
                        required_fields = ['documentName', 'revDate', 'RevNo', 'reviewDate', 'chapter', 'section', 'content']
                        for field in required_fields:
                            if field not in flattened_json:
                                flattened_json[field] = ""
                                print(f"   Added missing field '{field}' with empty string")

                        # If content is still missing, try to extract it from the original markdown
                        if not flattened_json.get('content'):
                            # Look for tables in the original markdown
                            content_lines = md_content.split('\n')
                            table_content = '\n'.join([line for line in content_lines if line.strip().startswith('|')])
                            if table_content:
                                flattened_json['content'] = table_content
                            else:
                                # Use a portion of the original text as a last resort
                                flattened_json['content'] = md_content[:500] + "..."
                        
                        print(f"   Successfully extracted and flattened JSON from markdown for page {page_num}")
                        enhanced_data.append(flattened_json)
                        continue  # Skip LLM processing for this page
                except json.JSONDecodeError:
                    print(f"   Failed to parse embedded JSON in markdown for page {page_num}. Will process with LLM.")
            
            # If we reach here, we need to process with LLM
            item_info = {'file': file_basename, 'page': page_num}
            tasks.append(extract_json_with_llm_async(md_content, model, semaphore, item_info))
            
        # Process remaining pages with LLM concurrently
        if tasks:
            print(f"   Processing {len(tasks)} pages with LLM concurrently...")
            results = await asyncio.gather(*tasks)
            
            # Add valid results to enhanced_data
            valid_results = [result for result in results if result is not None]
            enhanced_data.extend(valid_results)
            
            print(f"   Successfully processed {len(valid_results)} of {len(tasks)} pages with LLM.")
        
        total_processed = len(enhanced_data)
        print(f"   Total structured data extracted: {total_processed} pages")

        # Save the enhanced data back to the same file
        if enhanced_data:
            with open(input_filepath, 'w', encoding='utf-8') as outfile:
                json.dump(enhanced_data, outfile, indent=2, ensure_ascii=False)
            print(f"   Successfully saved enhanced data back to: {input_filepath}")
            processed_successfully = True
        else:
            print(f"   No enhanced data to save for {file_basename}. Skipping file save.")

    except json.JSONDecodeError as e:
        print(f"   Error decoding JSON from file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc(limit=1)
    except IOError as e:
        print(f"   Error reading/writing file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc(limit=1)
    except Exception as e:
        print(f"   An unexpected error occurred processing file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc()

    return processed_successfully

# --- Removed Summary Generation for File Functions ---

def process_file(input_filepath):
    """
    Reads a single JSON file, extracts 'page' and 'md' fields from the 'pages' list if needed,
    or validates that the existing structure is correct.
    """
    print(f"\n--- Processing file: {os.path.basename(input_filepath)} ---")
    extracted_data = []
    processed_successfully = False

    # Skip job_info.json files
    if os.path.basename(input_filepath) == 'job_info.json':
        print(f"   Skipping job_info.json file")
        return False

    try:
        # Check if the file exists before trying to open it
        if os.path.exists(input_filepath):
            # Open and read the input JSON file
            with open(input_filepath, 'r', encoding='utf-8') as f:
                # Load the JSON content from the file into a Python dictionary
                data = json.load(f)
            print(f"   Input file loaded: {os.path.basename(input_filepath)}")

            # Check if the data is already in the expected array format with page and md fields
            if isinstance(data, list) and all(isinstance(item, dict) and 'page' in item and 'md' in item for item in data):
                print(f"   File is already in the expected format with page and md fields. No extraction needed.")
                return True
            
            # If not in expected format, check if 'pages' key exists and is a list
            elif 'pages' in data and isinstance(data.get('pages'), list):
                # Iterate through each page in the 'pages' list
                for page_info in data.get("pages", []):
                    # Extract the page number and md content
                    page_number = page_info.get("page")
                    md_content = page_info.get("md")

                    # Add the extracted data to the list if both page number and md content exist
                    if page_number is not None and md_content is not None:
                        extracted_data.append({
                            "page": page_number,
                            "md": md_content
                        })
                    else:
                        print(f"   Warning: Missing 'page' or 'md' in a page entry within {os.path.basename(input_filepath)}.")

                print(f"   Extracted data for {len(extracted_data)} pages.")

                # --- Save the output file ---
                if extracted_data:  # Only save if we extracted something
                    # Save back to the same file
                    with open(input_filepath, 'w', encoding='utf-8') as outfile:
                        json.dump(extracted_data, outfile, indent=2, ensure_ascii=False)
                    print(f"   Successfully saved extracted data back to: {input_filepath}")
                    processed_successfully = True
                else:
                    print("   No data extracted, skipping file save.")
            else:
                print(f"   Warning: File format not recognized in {os.path.basename(input_filepath)}. Expected 'pages' list or array of page/md objects.")
                return False

        else:
            print(f"   Error: Input file path does not exist: {input_filepath}")

    except json.JSONDecodeError as e:
        print(f"   Error decoding JSON from file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc(limit=1)
    except IOError as e:
        print(f"   Error reading/writing file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc(limit=1)
    except Exception as e:
        print(f"   An unexpected error occurred processing file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc()

    return processed_successfully

# --- Removed Chunk Number Functions ---

async def main_async():
    """Main async function to process files with LLM"""
    print("Starting JSON processing with LLM enhancement...")
    print(f"Base Directory: {BASE_DIRECTORY}")
    
    all_files_count = 0
    success_count = 0
    error_count = 0
    
    # Create semaphore for limiting concurrent API calls
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Process each specified subfolder
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_DIRECTORY, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\nWarning: Subfolder not found: {subfolder_path}")
            continue
            
        print(f"\nProcessing files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder, excluding job_info.json
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        # Filter out job_info.json files
        json_files = [f for f in json_files if os.path.basename(f) != 'job_info.json']
        
        if not json_files:
            print(f"No JSON files found in '{subfolder_path}'.")
        else:
            print(f"Found {len(json_files)} JSON files to process in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    if await process_file_with_llm(file_path, semaphore=semaphore):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"--- CRITICAL ERROR processing file {os.path.basename(file_path)} ---")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    error_count += 1
    
    # Summary
    print(f"\n--- Processing Summary ---")
    print(f"Total files found: {all_files_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    
    print("\nScript finished.")

# --- Removed Summary Generation Main and Chunk Number Main Functions ---

# --- Main Script Logic ---
if __name__ == "__main__":
    print("Starting JSON processing pipeline...")
    print(f"Base Directory: {BASE_DIRECTORY}")
    
    # Option 1: Run the initial extraction only
    # Comment this out if already done and you only want to run LLM enhancement
    print("\n=== PHASE 1: Initial Extraction of Page and MD fields ===")
    all_files_count = 0
    success_count = 0
    error_count = 0
    
    # Process each specified subfolder
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_DIRECTORY, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\nWarning: Subfolder not found: {subfolder_path}")
            continue
            
        print(f"\nProcessing files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder, excluding job_info.json
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        # Filter out job_info.json files
        json_files = [f for f in json_files if os.path.basename(f) != 'job_info.json']
        
        if not json_files:
            print(f"No JSON files found in '{subfolder_path}'.")
        else:
            print(f"Found {len(json_files)} JSON files to process in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    if process_file(file_path):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"--- CRITICAL ERROR processing file {os.path.basename(file_path)} ---")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    error_count += 1
    
    # Summary for Phase 1
    print(f"\n--- Phase 1 Processing Summary ---")
    print(f"Total files found: {all_files_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    
    # Option 2: Run the LLM enhancement
    print("\n=== PHASE 2: LLM Enhancement for Structured Data Extraction ===")
    
    # Run the async main function for LLM enhancement
    asyncio.run(main_async()) 
    
    # Option 3: Generate summaries for processed files - moved to post_processing.py
    # Option 4: Add chunk numbers to all processed files - moved to post_processing.py 