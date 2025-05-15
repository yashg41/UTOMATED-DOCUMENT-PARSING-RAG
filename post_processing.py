import json
import os
import re
import glob
import traceback
import time
import asyncio
import random
import logging
from openai import AsyncOpenAI
import nest_asyncio

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
SUBFOLDERS = ['ISM Manual', 'OFFICE MANUAL','Policies','Sample Document Type']
# LLM Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 30
CONCURRENT_REQUESTS = 10  # Number of concurrent requests to OpenAI API
# --- ---

# --- Summary Generation Prompts ---
SUMMARY_PROMPT_TEMPLATE = """
You are an expert at summarizing structured text sections.

Summarize the following section by:
- Capturing **all relevant details, facts, and concepts** included in the section.
- Maintaining the **original meaning and scope** of the content without omitting information.
- Rephrasing or reorganizing the content into a **clear and coherent narrative**.
- Highlighting **key terms** or **important concepts** using Markdown formatting like **this**.

Section:
\"\"\"
{text}
\"\"\"

Comprehensive Summary:
"""

SHORT_SUMMARY_PROMPT_TEMPLATE = """
You are an expert at crafting clear, attention-grabbing section captions.

Generate a stylized, **one- or two-line** heading that summarizes the key topic of the following text:
- Capture the **core theme**, **purpose**, or **insight** of the section.
- Use **Markdown formatting** (e.g., `#`, `##`, `**bold**`, `*italics*`) to emphasize important terms.
- Keep the caption short — no more than **two lines**.
- Make it suitable for use as a section heading or title in structured notes.
- Output only the formatted heading — no explanations, comments, or extra text.

Text:
\"\"\"
{text}
\"\"\"

Highlighted Caption:
"""

# --- Function: Flatten Pages to Sections ---
def flatten_pages_to_sections(pages_data):
    """
    Transforms a list of page objects into a flat list of section objects.

    Each section object in the output list contains metadata from its parent page,
    its own section details (renamed), a constructed documentHeader, and fixed fields.

    Args:
        pages_data (list): A list of page dictionaries, processed to handle 'Continued' sections
                           and expecting keys like 'documentName', 'pageNumber', 'chapter', etc.

    Returns:
        list: A flat list of section dictionaries ready for final output.
    """
    flattened_list = []

    if not isinstance(pages_data, list):
        print("   Error: Input data to flatten_pages_to_sections must be a list.")
        return []

    for page_index, page in enumerate(pages_data):
        if not isinstance(page, dict):
            print(f"   Warning: Skipping invalid page data (not a dict) at index {page_index} during flattening.")
            continue

        page_num_for_log = page.get('pageNumber', f'index {page_index}')

        if 'sections' not in page or not isinstance(page['sections'], list):
            print(f"   Warning: Skipping page {page_num_for_log} due to missing or invalid 'sections' list during flattening.")
            continue

        # Extract page-level metadata (all keys except 'sections')
        page_metadata = {}
        for key, value in page.items():
            if key != 'sections':
                page_metadata[key] = value

        # Ensure documentName is the primary field
        if 'documentTitle' in page_metadata and 'documentName' not in page_metadata:
             page_metadata['documentName'] = page_metadata.pop('documentTitle')
        elif 'documentTitle' in page_metadata and 'documentName' in page_metadata:
             del page_metadata['documentTitle']

        # Iterate through sections within the current page
        for section_index, section in enumerate(page['sections']):
            if not isinstance(section, dict):
                print(f"   Warning: Skipping invalid section data (not a dict) on page {page_num_for_log}, section index {section_index} during flattening.")
                continue

            # Check essential section keys before proceeding
            if 'section_name' not in section or 'content' not in section:
                 print(f"   Warning: Skipping section on page {page_num_for_log}, index {section_index} due to missing 'section_name' or 'content'. Section data: {section}")
                 continue

            # Create a new dictionary for the flattened section
            flattened_section = {}

            # 1. Copy page-level metadata (includes the potentially updated chapter name)
            flattened_section.update(page_metadata)

            # 2. Copy section-specific data and rename keys
            section_name_raw = section.get('section_name')
            flattened_section['section'] = section_name_raw if section_name_raw is not None else 'N/A'
            flattened_section['originalText'] = section.get('content', '')

            # 3. Construct documentHeader
            doc_name = flattened_section.get('documentName', 'N/A')
            chapter_name = flattened_section.get('chapter', 'N/A')
            # Clean up chapter name if it contains newlines/excess whitespace
            if isinstance(chapter_name, str):
                chapter_name = re.sub(r'\s+', ' ', chapter_name).strip()
            section_name_header = flattened_section.get('section', 'N/A')
            page_num_header = flattened_section.get('pageNumber', 'N/A')

            document_header = (
                f"The document name: {doc_name} | "
                f"Chapter name: {chapter_name} | "
                f"Section name: {section_name_header} | "
                f"Page number: {page_num_header}"
            )
            flattened_section['documentHeader'] = document_header

            # 4. Add fixed DOC and identifier fields
            flattened_section['DOC'] = "SMPL"
            flattened_section['identifier'] = "SMPL"

            # 5. Final key renaming/cleanup: Rename 'pageNumber' back to 'page'
            if 'pageNumber' in flattened_section:
                flattened_section['page'] = flattened_section.pop('pageNumber')
            elif 'page' not in flattened_section:
                 flattened_section['page'] = page_num_header if page_num_header != 'N/A' else None

            if 'section_name' in flattened_section: del flattened_section['section_name']
            if 'content' in flattened_section: del flattened_section['content']

            flattened_list.append(flattened_section)

    return flattened_list
# --- END: Flattening Function ---

# --- Function: Transform Data to Include Sections ---
def transform_data_to_include_sections(input_data):
    """
    Transforms data from the format produced by llamaparse_processing.py to include sections.
    
    Args:
        input_data (list): A list of page dictionaries from llamaparse_processing.py output
        
    Returns:
        list: A list of page dictionaries with sections formatted correctly
    """
    transformed_pages = []

    for page in input_data:
        if not isinstance(page, dict):
            continue

        # Create the page object with all the page-level metadata
        page_obj = {k: v for k, v in page.items() if k != 'content' and k != 'section'}
        
        # Create sections list - initialize with one section using section and content from page
        sections = [{
            'section_name': page.get('section', ''),
            'content': page.get('content', '')
        }]
        
        # Add the sections to the page object
        page_obj['sections'] = sections
        
        # Add to the transformed pages list
        transformed_pages.append(page_obj)

    return transformed_pages

# --- Helper function for summary generation ---
def clean_text_for_llm(text: str) -> str:
    """Clean and prepare text for sending to LLM."""
    if not text:
        return ""
    # Remove excessive whitespace and normalize
    cleaned = " ".join(text.split())
    return cleaned

# --- Summary Generation Functions ---
async def generate_summaries_async(
    content: str,
    semaphore: asyncio.Semaphore,
    item_info: dict,  # Pass file/page info for logging
    async_client: AsyncOpenAI
) -> dict:
    """Generate summary and short summary from content text"""
    result = {
        "summary": "",
        "shortSummary": "" 
    }
    
    item_log_prefix = f"[File: {item_info.get('file', 'N/A')}, Page: {item_info.get('page', 'N/A')}]"
    cleaned_content = clean_text_for_llm(content)
    
    if not cleaned_content:
        print(f"{item_log_prefix} Content is empty. Skipping summary generation.")
        result["summary"] = "N/A - Empty Content"
        result["shortSummary"] = "N/A - Empty Content"
        return result

    # First generate comprehensive summary
    async with semaphore:
        try:
            print(f"{item_log_prefix} Generating comprehensive summary...")
            summary_response = await async_client.chat.completions.create(
                model="gpt-4o",  # Can customize this or use another model
                messages=[
                    {"role": "system", "content": "You are an expert summarizer specialized in generating complete, faithful overviews of document sections."},
                    {"role": "user", "content": SUMMARY_PROMPT_TEMPLATE.format(text=cleaned_content)}
                ],
                temperature=0.3,
            )
            
            summary = summary_response.choices[0].message.content.strip()
            result["summary"] = summary
            print(f"{item_log_prefix} Summary generated (length: {len(summary)})")
            
            # Now generate short summary based on the comprehensive summary
            print(f"{item_log_prefix} Generating short summary...")
            short_summary_response = await async_client.chat.completions.create(
                model="gpt-4o",  # Can customize this or use another model
                messages=[
                    {"role": "system", "content": "You are an expert at creating impactful, well-formatted section headings."},
                    {"role": "user", "content": SHORT_SUMMARY_PROMPT_TEMPLATE.format(text=summary)}
                ],
                temperature=0.3,
            )
            
            short_summary = short_summary_response.choices[0].message.content.strip()
            # Limit to at most 2 lines
            short_summary_lines = short_summary.split('\n')
            result["shortSummary"] = '\n'.join(short_summary_lines[:2])
            print(f"{item_log_prefix} Short summary generated (length: {len(result['shortSummary'])})")
            
        except Exception as e:
            print(f"{item_log_prefix} Error generating summaries: {str(e)}")
            result["summary"] = f"Error generating summary: {str(e)[:100]}..."
            result["shortSummary"] = "Error generating short summary"
    
    return result

# --- Function: Generate summaries for file ---
async def generate_summaries_for_file(input_filepath, semaphore=None):
    """
    Reads a JSON file containing structured data, generates summaries for each item,
    and saves the result back to the same file.
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(3)  # Default to 3 concurrent requests
        
    print(f"\n--- Generating summaries for file: {os.path.basename(input_filepath)} ---")
    processed_successfully = False

    # Skip job_info.json files
    if os.path.basename(input_filepath) == 'job_info.json':
        print(f"   Skipping job_info.json file")
        return False

    # Initialize the OpenAI client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment variables. Skipping summary generation.")
        return False
    async_client = AsyncOpenAI(api_key=openai_api_key)

    try:
        # Check if the file exists before trying to open it
        if not os.path.exists(input_filepath):
            print(f"   Error: Input file path does not exist: {input_filepath}")
            return False

        # Open and read the input JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Input file loaded: {os.path.basename(input_filepath)}")

        # Validate the data structure
        if not isinstance(data, list):
            print(f"   Warning: File {os.path.basename(input_filepath)} does not contain a list. Skipping.")
            return False

        # Process each item to generate summaries
        file_basename = os.path.basename(input_filepath)
        summary_tasks = []
        skipped_items = 0
        item_indices = []
        
        # Create tasks for summary generation
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"   Warning: Item {i} is not a dictionary. Skipping this item.")
                continue
            
            # Check if summary and shortSummary already exist
            if 'summary' in item and item['summary'] and 'shortSummary' in item and item['shortSummary']:
                print(f"   Item {i} already has summary and shortSummary. Skipping summary generation.")
                skipped_items += 1
                continue
                
            if 'originalText' not in item or not item['originalText']:
                if 'content' not in item or not item['content']:
                    print(f"   Warning: Empty or missing 'originalText' and 'content' field for item {i}. Skipping summary generation.")
                    continue
                else:
                    content = item['content']
            else:
                content = item['originalText']
            
            item_info = {'file': file_basename, 'page': item.get('page', i+1)}
            summary_tasks.append(generate_summaries_async(content, semaphore, item_info, async_client))
            item_indices.append(i)
        
        if not summary_tasks:
            if skipped_items > 0:
                print(f"   All {skipped_items} items already have summaries in {file_basename}. No new summaries needed.")
                return True
            else:
                print(f"   No items with valid content found in {file_basename}. Skipping.")
                return False
            
        # Process all items concurrently
        print(f"   Generating summaries for {len(summary_tasks)} items concurrently (skipped {skipped_items} existing summaries)...")
        
        # Use asyncio.gather to truly run all tasks concurrently
        summary_results = await asyncio.gather(*summary_tasks)
        
        # Update the data with the results
        for idx, result in zip(item_indices, summary_results):
            if result:
                data[idx]['summary'] = result['summary']
                data[idx]['shortSummary'] = result['shortSummary']
        
        print(f"   Generated summaries for {len(summary_tasks)} items and kept {skipped_items} existing summaries.")

        # Save the enhanced data back to the same file
        with open(input_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=2, ensure_ascii=False)
        print(f"   Successfully saved data with summaries back to: {input_filepath}")
        processed_successfully = True

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

async def generate_summaries_main_async():
    """Main async function to generate summaries for processed JSON files"""
    print("Starting summary generation for processed JSON files...")
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
            
        print(f"\nGenerating summaries for files in subfolder: {subfolder}")
        
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
                    if await generate_summaries_for_file(file_path, semaphore=semaphore):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"--- CRITICAL ERROR generating summaries for file {os.path.basename(file_path)} ---")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    error_count += 1
    
    # Summary
    print(f"\n--- Summary Generation Summary ---")
    print(f"Total files found: {all_files_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    
    print("\nSummary generation finished.")

# --- Chunk Number Functions ---
def add_chunk_numbers(input_filepath):
    """
    Adds a 'chunkNo' field to each item in a JSON list file.
    
    Args:
        input_filepath (str): Path to the JSON file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n--- Adding chunk numbers to file: {os.path.basename(input_filepath)} ---")
    
    # Skip job_info.json files
    if os.path.basename(input_filepath) == 'job_info.json':
        print(f"   Skipping job_info.json file")
        return False
        
    try:
        # Check if the file exists
        if not os.path.exists(input_filepath):
            print(f"   Error: Input file path does not exist: {input_filepath}")
            return False
            
        # Read the JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check if data is a list
        if not isinstance(data, list):
            print(f"   Error: File {os.path.basename(input_filepath)} does not contain a JSON list. Skipping.")
            return False
        
        # Check if all items already have the same chunk number (possible bug)
        chunk_numbers = set()
        for item in data:
            if isinstance(item, dict) and 'chunkNo' in item:
                chunk_numbers.add(item['chunkNo'])
        
        # If all items have the same chunk number, we need to fix it
        if len(chunk_numbers) == 1 and len(data) > 1:
            print(f"   Found {len(data)} items with identical chunk number {list(chunk_numbers)[0]}. Fixing...")
            
            # Add the chunkNo field to each item with sequential numbering
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Update to sequential chunk number
                    item['chunkNo'] = i + 1
                    
            # Save the updated data back to the file
            with open(input_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"   Successfully fixed chunk numbers in {os.path.basename(input_filepath)}")
            return True
            
        # Add the chunkNo field to each item
        updated_data = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Create a new dictionary with chunkNo at the beginning
                updated_item = {'chunkNo': i + 1}
                updated_item.update(item)  # Add existing key-value pairs
                updated_data.append(updated_item)
            else:
                print(f"   Warning: Item at index {i} is not a dictionary. Keeping original structure.")
                updated_data.append(item)
                
        # Save the updated data back to the file
        with open(input_filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
        print(f"   Successfully added 'chunkNo' field to {len(updated_data)} items in {os.path.basename(input_filepath)}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   Error decoding JSON from file {os.path.basename(input_filepath)}: {e}")
        return False
    except Exception as e:
        print(f"   Error processing file {os.path.basename(input_filepath)}: {e}")
        traceback.print_exc(limit=2)
        return False

async def process_chunk_numbers_main_async():
    """Main async function to add chunk numbers to all processed files"""
    print("Starting chunk number addition for processed JSON files...")
    print(f"Base Directory: {BASE_DIRECTORY}")
    
    all_files_count = 0
    success_count = 0
    error_count = 0
    
    # Process each specified subfolder
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_DIRECTORY, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\nWarning: Subfolder not found: {subfolder_path}")
            continue
            
        print(f"\nAdding chunk numbers to files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder, excluding job_info.json
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        json_files = [f for f in json_files if os.path.basename(f) != 'job_info.json']
        
        if not json_files:
            print(f"No JSON files found in '{subfolder_path}'.")
        else:
            print(f"Found {len(json_files)} JSON files to process in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    if add_chunk_numbers(file_path):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"--- CRITICAL ERROR adding chunk numbers to file {os.path.basename(file_path)} ---")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    error_count += 1
    
    # Summary
    print(f"\n--- Chunk Number Addition Summary ---")
    print(f"Total files found: {all_files_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    
    print("\nChunk number addition finished.")

def fix_identical_chunk_numbers():
    """
    Dedicated function to find and fix JSON files where all items have the same chunk number.
    This can be run as a standalone process.
    """
    print("\n=== Starting process to fix identical chunk numbers ===")
    print(f"Base Directory: {BASE_DIRECTORY}")
    
    all_files_count = 0
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each specified subfolder
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_DIRECTORY, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\nWarning: Subfolder not found: {subfolder_path}")
            continue
            
        print(f"\nScanning files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder, excluding job_info.json
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        json_files = [f for f in json_files if os.path.basename(f) != 'job_info.json']
        
        if not json_files:
            print(f"No JSON files found in '{subfolder_path}'.")
        else:
            print(f"Found {len(json_files)} JSON files to scan in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    filename = os.path.basename(file_path)
                    # Read the JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Skip if not a list
                    if not isinstance(data, list):
                        print(f"   Skipping {filename}: Not a JSON list")
                        skipped_count += 1
                        continue
                    
                    # Skip if empty list
                    if not data:
                        print(f"   Skipping {filename}: Empty JSON list")
                        skipped_count += 1
                        continue
                    
                    # Check if all items have chunk numbers
                    has_chunk_numbers = all('chunkNo' in item for item in data if isinstance(item, dict))
                    if not has_chunk_numbers:
                        print(f"   Skipping {filename}: Some items missing chunkNo field")
                        skipped_count += 1
                        continue
                    
                    # Check if all items have the same chunk number
                    chunk_numbers = set(item['chunkNo'] for item in data if isinstance(item, dict) and 'chunkNo' in item)
                    
                    if len(chunk_numbers) == 1 and len(data) > 1:
                        print(f"   {filename}: Found {len(data)} items with identical chunk number {list(chunk_numbers)[0]}. Fixing...")
                        
                        # Update each item with sequential chunk numbers
                        for i, item in enumerate(data):
                            if isinstance(item, dict):
                                item['chunkNo'] = i + 1
                        
                        # Save the updated data back to the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        
                        print(f"   Successfully fixed chunk numbers in {filename}")
                        fixed_count += 1
                    else:
                        print(f"   {filename}: No fix needed. Found {len(chunk_numbers)} different chunk numbers.")
                        skipped_count += 1
                
                except json.JSONDecodeError as e:
                    print(f"   Error parsing JSON file {filename}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"   Error processing file {filename}: {e}")
                    traceback.print_exc(limit=2)
                    error_count += 1
    
    # Summary
    print(f"\n--- Fix Identical Chunk Numbers Summary ---")
    print(f"Total files scanned: {all_files_count}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files skipped (no issues): {skipped_count}")
    print(f"Files with errors: {error_count}")
    
    print("\nProcess completed.")
    
    return fixed_count > 0

# --- Function: Process Single Input JSON File ---
def process_single_json_file(input_filepath):
    """
    Loads a JSON file, handles 'Continued' chapters, merges 'Continued' sections, 
    flattens the structure, adds/renames fields, and replaces the original file.
    """
    input_basename = os.path.basename(input_filepath)

    print(f"\n--- Processing file: {input_basename} ---")

    loaded_pages = None

    try:
        # 1. Load the main JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if the file is already correctly structured with chunk numbers
        if isinstance(data, list) and all(isinstance(item, dict) and 'chunkNo' in item for item in data):
            print(f"   File {input_basename} already has chunk numbers. Preserving existing structure.")
            return

        # 2. Validate Input Structure and get page list
        if isinstance(data, list):
            loaded_pages = data
        elif isinstance(data, dict) and 'pages' in data and isinstance(data['pages'], list):
            loaded_pages = data['pages']
            print(f"   Input is a dictionary. Using {len(loaded_pages)} page entries from 'pages' key.")
        else:
            raise ValueError("Input JSON structure is not a list of pages nor a dict containing a 'pages' list.")

        if not loaded_pages:
             print("   Warning: Input file contains no page entries. Skipping processing.")
             return

        # Extract input filename without extension for use in documentName if needed
        input_name_part, _ = os.path.splitext(input_basename)

        # 3. Convert data from llamaparse_processing.py format to include sections
        pre_processed_pages = transform_data_to_include_sections(loaded_pages)
             
        # 4. Pre-process Pages: Handle "Continued" Chapters and Prepare Keys
        print(f"   Pre-processing {len(pre_processed_pages)} pages (handling chapters, keys)...")
        processed_pages = []
        last_valid_chapter_name = None

        for i, page_data in enumerate(pre_processed_pages):
             # Basic validation of page structure
            if not isinstance(page_data, dict):
                print(f"   Warning: Skipping invalid page entry (not a dict) at index {i}.")
                continue

            current_page_num = page_data.get('page', f'index {i}')

            # --- START: Chapter Handling ---
            current_chapter_name = page_data.get('chapter')

            if isinstance(current_chapter_name, str):
                chapter_stripped = current_chapter_name.strip()
                chapter_stripped_lower = chapter_stripped.lower()
                if chapter_stripped_lower == 'continued':
                    # Replace "Continued" with the last valid chapter name seen
                    replacement_chapter = last_valid_chapter_name if last_valid_chapter_name else 'N/A'
                    print(f"      Info: Replacing 'Continued' chapter on page {current_page_num} with '{replacement_chapter}'.")
                    page_data['chapter'] = replacement_chapter
                elif chapter_stripped:
                    # It's a valid new chapter name, update the tracker
                    last_valid_chapter_name = chapter_stripped
                    page_data['chapter'] = last_valid_chapter_name
                else:
                    # Chapter is empty string or whitespace after stripping
                    replacement_chapter = last_valid_chapter_name if last_valid_chapter_name else 'N/A'
                    print(f"      Warning: Chapter is empty/whitespace on page {current_page_num}. Using last valid: '{replacement_chapter}'.")
                    page_data['chapter'] = replacement_chapter
            elif current_chapter_name is None:
                # Chapter key is missing or None
                replacement_chapter = last_valid_chapter_name if last_valid_chapter_name else 'N/A'
                print(f"      Warning: Chapter is missing on page {current_page_num}. Using last valid: '{replacement_chapter}'.")
                page_data['chapter'] = replacement_chapter
            else:
                # Chapter is not a string (e.g., number, bool) - treat as invalid
                replacement_chapter = last_valid_chapter_name if last_valid_chapter_name else 'N/A'
                print(f"      Warning: Chapter is not a string ('{current_chapter_name}') on page {current_page_num}. Using last valid: '{replacement_chapter}'.")
                page_data['chapter'] = replacement_chapter
            # --- END: Chapter Handling ---

            # --- Prepare other keys for flattening ---
            # Rename 'page' to 'pageNumber'
            if 'page' in page_data:
                page_data['pageNumber'] = page_data.pop('page')
            elif 'pageNumber' not in page_data:
                 page_num_val = None
                 if isinstance(current_page_num, (int, float)): page_num_val = int(current_page_num)
                 elif isinstance(current_page_num, str) and current_page_num.isdigit(): page_num_val = int(current_page_num)
                 else: page_num_val = f'unknown_{i}'
                 page_data['pageNumber'] = page_num_val

            # Ensure 'documentName', copying from 'documentTitle' if needed
            doc_name_found = False
            if 'documentName' in page_data and page_data['documentName']:
                 doc_name_found = True
            elif 'documentTitle' in page_data and page_data['documentTitle']:
                page_data['documentName'] = page_data.pop('documentTitle')
                doc_name_found = True
            elif 'documentTitle' in page_data:
                del page_data['documentTitle']

            if not doc_name_found:
                 page_data['documentName'] = input_name_part.replace('_', ' ').title()
                 print(f"      Warning: No documentName/Title found on page {page_data.get('pageNumber', 'N/A')}. Using filename: '{page_data['documentName']}'.")

            # Ensure other common keys are present or defaulted
            page_data.setdefault('revDate', 'N/A')
            page_data.setdefault('RevNo', 'N/A')
            page_data.setdefault('reviewDate', 'N/A')

            # Ensure 'sections' exists (initialize if missing)
            page_data.setdefault('sections', [])
            if not isinstance(page_data['sections'], list):
                 print(f"   Warning: Invalid 'sections' format (not a list) on page {page_data.get('pageNumber', 'N/A')} before merging. Resetting to empty list.")
                 page_data['sections'] = []

            processed_pages.append(page_data)

        # 5. Merge "Continued" Sections
        print(f"   Merging 'Continued' sections...")
        merged_pages = []
        last_section_data_ref = None

        for i, page_data in enumerate(processed_pages):
            current_page_num_for_log = page_data.get('pageNumber', f'index {i}')
            sections_in_page = page_data.get('sections', [])

            current_page_final_sections = []
            for j, section_data in enumerate(sections_in_page):
                if not isinstance(section_data, dict):
                    print(f"   Warning: Skipping invalid section item (not a dict) on page {current_page_num_for_log}, section index {j} during merge.")
                    continue

                if 'section_name' not in section_data or 'content' not in section_data:
                     print(f"   Warning: Skipping section on page {current_page_num_for_log}, index {j} during merge due to missing keys. Data: {section_data}")
                     continue

                section_name_raw = section_data.get('section_name')
                section_name_stripped_lower = section_name_raw.strip().lower() if isinstance(section_name_raw, str) else ""
                section_content = section_data.get('content', '')
                if not isinstance(section_content, str): section_content = str(section_content)

                if section_name_stripped_lower == 'continued':
                    if last_section_data_ref is not None and 'content' in last_section_data_ref:
                        separator = "\n\n"
                        prev_content = last_section_data_ref.get('content', '')
                        if not isinstance(prev_content, str): prev_content = str(prev_content)
                        last_section_data_ref['content'] = prev_content + separator + section_content
                    else:
                        print(f"      Warning: Found 'Continued' section (page {current_page_num_for_log}, index {j}) with no valid preceding section. Creating 'Orphaned Continued' section.")
                        section_data['section_name'] = f"Orphaned Continued Section (Page {current_page_num_for_log})"
                        current_page_final_sections.append(section_data)
                        last_section_data_ref = section_data
                else:
                    current_page_final_sections.append(section_data)
                    last_section_data_ref = section_data

            if current_page_final_sections:
                page_data['sections'] = current_page_final_sections
                merged_pages.append(page_data)
            else:
                 print(f"   Note: Page {current_page_num_for_log} resulted in zero sections after merging 'Continued'. Discarding page.")

        print(f"   Finished merging 'Continued' sections. Resulting page count for flattening: {len(merged_pages)}")

        # 6. Flatten the processed & merged pages data
        flattened_data = flatten_pages_to_sections(merged_pages)

        # Add chunk numbers right after flattening
        if flattened_data:
            print(f"   Adding chunk numbers to {len(flattened_data)} items...")
            updated_data = []
            for i, item in enumerate(flattened_data):
                if isinstance(item, dict):
                    # Create a new dictionary with chunkNo at the beginning
                    updated_item = {'chunkNo': i + 1}
                    # Add existing key-value pairs
                    for key, value in item.items():
                        updated_item[key] = value
                    updated_data.append(updated_item)
                else:
                    print(f"   Warning: Item at index {i} is not a dictionary. Keeping original structure.")
                    updated_data.append(item)
            flattened_data = updated_data
            print(f"   Successfully added chunk numbers to {len(flattened_data)} items")

        # 7. Replace the original file with the processed data
        if flattened_data:
            print(f"   Saving {len(flattened_data)} processed sections back to original file: {input_filepath}")
            with open(input_filepath, 'w', encoding='utf-8') as f_original:
                json.dump(flattened_data, f_original, indent=4, ensure_ascii=False)
            print(f"   Successfully replaced original file with processed data.")
        else:
            print("   No sections were generated after processing and flattening; not modifying original file.")

    except FileNotFoundError:
        print(f"   ERROR: Input file not found at '{input_filepath}'")
    except json.JSONDecodeError as e:
        print(f"   ERROR: Could not decode the main JSON file '{input_basename}'. It might be corrupted. Error at char {e.pos}: {e.msg}")
    except ValueError as e:
         print(f"   ERROR: Invalid JSON structure in file '{input_basename}'. {e}")
    except Exception as e:
        print(f"   ERROR: An unexpected error occurred processing file '{input_basename}': {type(e).__name__} - {e}")
        traceback.print_exc()

# --- Main Script Execution ---
def main():
    print("Starting JSON post-processing script...")
    print(f"Base Directory: {BASE_DIRECTORY}")
    
    all_files_count = 0
    success_count = 0
    error_count = 0
    
    start_time = time.time()
    
    # Process each specified subfolder
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_DIRECTORY, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\nWarning: Subfolder not found: {subfolder_path}")
            continue
            
        print(f"\nProcessing files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder, excluding job_info.json
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        json_files = [f for f in json_files if os.path.basename(f) != 'job_info.json']
        
        if not json_files:
            print(f"No JSON files found in '{subfolder_path}'.")
        else:
            print(f"Found {len(json_files)} JSON files to process in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    process_single_json_file(file_path)
                    success_count += 1
                except Exception as e:
                    print(f"--- CRITICAL ERROR processing file {os.path.basename(file_path)} ---")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    error_count += 1
    
    end_time = time.time()
    
    # Fix any files with identical chunk numbers
    print("\nChecking for files with identical chunk numbers...")
    fix_identical_chunk_numbers()
    
    # Summary
    print(f"\n--- Processing Summary ---")
    print(f"Total files processed: {all_files_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {error_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    asyncio.run(process_chunk_numbers_main_async()) 