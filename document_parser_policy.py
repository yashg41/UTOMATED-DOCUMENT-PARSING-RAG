#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policies Document Parser Module

This module handles parsing of Policies using LlamaIndex.
"""

import os
import time
import json
import asyncio
import requests
import traceback
import nest_asyncio
import glob
from dotenv import load_dotenv
from llama_parse import LlamaParse
from typing import Dict, List, Optional, Any, Tuple

# Apply the nest_asyncio patch to allow nested asyncio loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# System prompt for Ploicy document parsing
SYSTEM_PROMPT = """
You are an expert document parsing assistant. Your task is to convert the  document into a structured JSON object, paying meticulous attention to section boundaries, content association, and page continuity.

**Instructions:**

1.  **Parse Top-Level Metadata:**
    * Extract: `documentTitle` (top of page), `revDate` (e.g., "Revision Date", "Date"), `RevNo` ("Revision No.", "Rev", "Issue"), and `reviewDate` ("Review Date", "Next Review Date"). Format as YYYY-MM-DD where possible. Use `null` if missing.
    * Identify the main `chapter` identifier (e.g., ("Chapter X", "SECTION Y") followed by the title of chapter or clear heading like "1. General Information").
    * In some cases, chapter name and document title can be same.
    * If the page clearly starts mid-chapter ie no chapter title present, set `chapter` to `"Continued"`/ "Null" whichever is appropriate.

2.  **Identify Section Boundaries and Content:**
    * Scan the page for distinct section headers (e.g., "1.1 Section Title", "Appendix A", "Section 2B","Table of Contents"). These often have unique formatting (bolding, larger font, numbering) or look different from regular text.
    * **Crucially, a section's content includes EVERYTHING from its distinct header up to, the *next* distinct section header found in the future page.**
    * This means:
        * All text paragraphs, lists, etc..
        * Tables (formatted as HTML) and Figure Descriptions (see point 3) appearing between two section headers are part of the *first* section's content.
        * **Specifically address text after complex elements:** Any text, lists, or paragraphs appearing *after* a table or figure description, but *before* the next distinct section header, MUST be appended to the `content` of the current section. IF not create a new section for such trailing content and put the section name as "Continued".
    * **Handling the Start of the Page Content (Be Vigilant for Continuations):**
        When examining the very first block of content on the page:
          * If it clearly starts with a distinct section header that matches the document's established pattern (e.g., "2A.3 RELATED PROCEDURES", "Appendix B"), assign that header as the `section_name`.
          * **However, if it lacks such a clear, distinct header:**
              * **Strongly prefer** assigning `section_name: "Continued"` if the text appears to pick up mid-thought (e.g., starts mid-sentence, starts with lowercase letters unless clearly a list item), or if it logically follows the typical flow from where the previous page likely ended.
              * Be **very cautious** about interpreting minor formatting changes (like an initial bold word that isn't numbered like other headers) as the start of a new section. Check if it aligns with the document's overall section heading style.
          * **In cases of ambiguity at the page start: If it's not definitively a new section start, default to marking it as `section_name: "Null"`.**
        The `"Continued"` name is strictly reserved for the *start* of the page if the text appears to pick up mid-thought or if it logically follows the typical flow from where the previous page likely ended

3.  **Extract and Describe Content Elements:**
    * Extract all textual content. Maintain paragraph structure.
    * Represent tables using standard HTML tags (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`). Include table content accurately.
    * **For figures, images, or diagrams:** Describe based on visual analysis and context from surrounding text using the format `"Figure Description: [Your detailed description here]"`.
        * **Identify Type:** Start by stating the type of visual (e.g., "Flowchart:", "Bar graph:", "Photograph:", "Technical drawing:", "Illustration:").
        * **Describe Content Thoroughly:** Detail the main subject, **all visible text including labels, annotations, and data points**,mention **exact data**, trends, or key comparisons shown), symbols and their meanings **within the context**, relationships depicted (e.g., connections in flowcharts, hierarchies in diagrams), significant colors if they convey meaning, and the overall composition or layout. For photos, describe the scene, objects, people (if depicted, describe neutrally and factually based on visual cues), and setting realistically and completely.
        **Be Specific & Accurate:** Ensure all details present in the visual are described.
	**Transcribe text within the image exactly as it appears.** Use quantifiable descriptions where appropriate (e.g., "shows a 3-stage process", "contains 5 columns labeled...").
    * **Crucially, do NOT treat figure captions or titles as section headers.** They are part of the figure's descriptive context or textual content.

4.  **Handle End-of-Page Continuation:**
    * Examine the very *last* section object created for the page.
    * If its `content` seems to end abruptly mid-sentence or mid-paragraph right at the end of the page's main content area (before any footer), add the flag `"continues_on_next_page": true` to this last section object.
    * Otherwise, do not add this flag.

5.  **Exclude Headers and Footers:**
    * Identify and ignore repetitive page headers (e.g., repeating document titles/codes) and footers (e.g., page numbers, "Confidential", date stamps unrelated to revision). Do not include these in the `content` fields.

6.  **Structure the JSON Output:**
    * Produce a single JSON object per page.
    * Include the top-level fields: `documentTitle`, `revDate`, `RevNo`, `reviewDate`, `chapter`.
    * Include a top-level array named `sections`.
    * Each element in the `sections` array must be an object with:
        * `section_name`: The extracted section header (e.g., "2A.1 GENERAL") or `"Continued"` or Null (only if applicable per Rule 2).
        * `content`: A string containing all extracted text, HTML tables, and figure descriptions belonging to that section, in the correct order.
        * `continues_on_next_page`: (Optional) A boolean flag, only present and set to `true` on the *last* section object if it meets the criteria in Rule 4.

7. DO NOT MISS ANYTHING FROM THE PAGE.

**Example JSON Structure:**

```json
{
  "documentTitle": "HEALTH AND SAFETY MANUAL",
  "revDate": "2022-12-15",
  "RevNo": "0",
  "reviewDate": "2025-01-24",
  "chapter": "Chapter: 2A\\\\nPERMIT TO WORK - GENERAL", // Or "Continued"
  "sections": [
    {
      "section_name": "2A.1 GENERAL", // Or "Continued" if page starts mid-section
      "content": "Initial text of the section...\\\\n\\\\n<table>...HTML Table Content...</table>\\\\n\\\\nThis text appears AFTER the table but BEFORE section 2A.2, so it is INCLUDED here.\\\\n\\\\nFigure Description:[]
    },
    {
      "section_name": "2A.2 DESCRIPTION OF PTW SYSTEM",
      "content": "Text content for section 2A.2. This section might be very long and end abruptly at the bottom of the page...",
      "continues_on_next_page": true // Added because content seems cut off by page end
    }
    // Potentially more sections if they fit on the page
  ]
}
// --- Example: Next Page ---
{
  "documentTitle": "HEALTH AND SAFETY MANUAL",
  "revDate": "2022-12-15",
  "RevNo": "0",
  "reviewDate": "2025-01-24",
  "chapter": "Continued", // Assuming chapter started on previous page
  "sections": [
    {
      "section_name": "Continued", // Because page starts mid-section 2A.2 (and lacks a clear new header)
      "content": "...continuation of the text from section 2A.2 from the previous page. Now finishing the section content here."
      // No 'continues_on_next_page' flag needed if this section finishes here
    },
    {
      "section_name": "2A.3 RELATED PROCEDURES",
      "content": "Content for the next section..."
    }
  ]
}
"""

# --- Configuration ---
# First check if env vars are set
env_llama_key = os.getenv("LLAMA_API_KEY")
env_vendor_key = os.getenv("VENDOR_API_KEY")

# Debug information about environment variables
print("---- API KEY DEBUG INFO ----")
print(f"LLAMA_API_KEY environment variable set: {'Yes' if env_llama_key else 'No'}")
print(f"VENDOR_API_KEY environment variable set: {'Yes' if env_vendor_key else 'No'}")
print(f".env file exists in current directory: {'Yes' if os.path.exists('.env') else 'No'}")
if env_llama_key:
    masked_api_key = f"{env_llama_key[:5]}...{env_llama_key[-4:]}" if len(env_llama_key) > 10 else "***masked***"
    print(f"Using LLAMA_API_KEY from environment: {masked_api_key}")
else:
    print("WARNING: LLAMA_API_KEY not found in environment variables!")

# Set variables with or without defaults
LLAMA_API_KEY = env_llama_key
VENDOR_API_KEY = env_vendor_key
VENDOR_MODEL = os.getenv("VENDOR_MODEL", "anthropic-sonnet-3.7")
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "3"))  # Default to 3
OUTPUT_SUFFIX = "_result"  # Suffix for output JSON files
print("---------------------------")

def submit_parsing_job_direct_api(pdf_path, api_key, vendor_api_key, vendor_model_name):
    """
    Submits the document parsing job directly via API using a Llama API key and vendor API key.
    Returns the job ID ('id') on success, or the HTTP status code (int)
    on specific API errors (401, 429), or None for other errors.
    Handles PDF files ONLY.
    """
    file_extension = os.path.splitext(pdf_path)[1].lower()
    file_name = os.path.basename(pdf_path)

    if file_extension != '.pdf':
        print(f" > [{file_name}] Error: Unsupported file type '{file_extension}'. Only PDF files are supported. Skipping.")
        return None # Indicate failure, not a specific API error code

    print(f"Submitting '{file_name}' via Llama Cloud API using {vendor_model_name} for  Policies parsing...")
    upload_url = "https://api.cloud.llamaindex.ai/api/parsing/upload"
    headers = {
        "Authorization": f"Bearer {api_key}", # Use the Llama API key
        "accept": "application/json",
    }

    # Prepare payload data
    multipart_payload_data = {
        'parsing_instruction': SYSTEM_PROMPT,
        'invalidate_cache': 'false',
        'use_vendor_multimodal_model': 'false',
        'vendor_multimodal_model_name': vendor_model_name,
        'vendor_api_key': vendor_api_key,  # Add the vendor API key
        'output_tables_as_html': 'true',
        'parse_mode' : "parse_page_with_lvm"
    }

    try:
        with open(pdf_path, 'rb') as f:
            files_payload = {
                'file': (file_name, f, 'application/pdf') # MIME type is fixed for PDF
            }
            for key, value in multipart_payload_data.items():
                files_payload[key] = (None, value)

            response = requests.post(upload_url, headers=headers, files=files_payload)

        print(f" > [{file_name}] API Submission Response Status: {response.status_code}")

        if response.status_code == 200:
            response_json = response.json()
            job_id = response_json.get("id")
            if job_id:
                print(f" > [{file_name}] Successfully submitted job. Job ID: {job_id}")
                return job_id # Return job ID string on success
            else:
                print(f" > [{file_name}] API response OK, but 'id' key not found: {response_json}")
                return None # Indicate other failure
        elif response.status_code in [401, 429]:
             # Return the specific status code for key-related errors
             if response.status_code == 429:
                 # Mask API keys for security when logging
                 masked_llama_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "***masked***"
                 masked_vendor_key = f"{vendor_api_key[:5]}...{vendor_api_key[-4:]}" if len(vendor_api_key) > 10 else "***masked***"
                 print(f" > [{file_name}] API rate limit exceeded (429). This indicates too many requests.")
                 print(f" > [{file_name}] Using Llama API key: {masked_llama_key}, Vendor API key: {masked_vendor_key}")
                 print(f" > [{file_name}] Consider implementing exponential backoff or reducing concurrent requests.")
             else:
                 print(f" > [{file_name}] API authentication failed (401). Check your API keys.")
             print(f" > [{file_name}] API submission failed: Status {response.status_code}.")
             return response.status_code
        elif response.status_code == 400:
             print(f" > [{file_name}] API submission failed: Bad Request (400). Check file/request details.")
             print(f" > Response: {response.text}")
             return None # Indicate other failure
        else:
            print(f" > [{file_name}] API submission failed. Status: {response.status_code}, Response: {response.text}")
            return None # Indicate other failure

    except requests.exceptions.RequestException as e:
        print(f" > [{file_name}] Error submitting job via API: {e}")
        return None # Indicate other failure
    except FileNotFoundError:
        print(f" > [{file_name}] Error: File not found at {pdf_path}")
        return None # Indicate other failure
    except Exception as e:
        print(f" > [{file_name}] An unexpected error occurred during submission: {e}")
        traceback.print_exc()
        return None # Indicate other failure

def get_json_results(job_id, api_key, pdf_filename, retries=10, delay=20):
    """
    Retrieve JSON results using the job ID and the specific API key it was submitted with.
    Handles retries for transient errors like PENDING or 5xx.
    Returns the JSON result on success, or None on failure.
    """
    if not job_id:
        print(f" > [{pdf_filename}] Error: Cannot retrieve JSON results without a job ID.")
        return None

    print(f" > [{pdf_filename}] Attempting to retrieve JSON results for job ID: {job_id}")
    result_url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/json"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}" # Use the SAME key used for submission
    }

    for attempt in range(retries):
        print(f" > [{pdf_filename}] JSON retrieval attempt {attempt + 1}/{retries} for job {job_id}...")
        try:
            response = requests.get(result_url, headers=headers)
            print(f" > [{pdf_filename}] JSON HTTP response status: {response.status_code}")

            if response.status_code == 200:
                try:
                    result_json = response.json()
                    if isinstance(result_json, dict) and result_json.get("status") == "ERROR":
                         print(f" > [{pdf_filename}] Job {job_id} completed with ERROR status (JSON endpoint).")
                         print(f" > [{pdf_filename}] Error details: {result_json.get('error_message', 'No details provided.')}")
                         return None # Job failed server-side

                    if isinstance(result_json, dict) and result_json.get("status") == "PENDING":
                         print(f" > [{pdf_filename}] Job {job_id} is still PENDING.")
                         # Fall through to retry logic

                    elif isinstance(result_json, (dict, list)):
                        print(f" > [{pdf_filename}] Successfully retrieved and parsed JSON results for job {job_id}.")
                        return result_json
                    else:
                        print(f" > [{pdf_filename}] Unexpected JSON result format for job {job_id}: {type(result_json)}")
                        print(f" > Content sample: {str(result_json)[:200]}")
                        # Treat as potentially not ready yet, fall through to retry logic
                except json.JSONDecodeError as e:
                     print(f" > [{pdf_filename}] Error decoding JSON response for job {job_id}: {e}")
                     print(f" > Response text sample: {response.text[:200]}")
                     # Potentially temporary, fall through to retry logic

            elif response.status_code == 404:
                 print(f" > [{pdf_filename}] JSON result not found or job {job_id} not ready yet (404).")
            elif response.status_code == 401:
                 # This is a fatal error for THIS job retrieval attempt.
                 print(f" > [{pdf_filename}] JSON retrieval failed: Unauthorized (401) for job {job_id}. Stopping retries for this file.")
                 return None
            elif response.status_code == 429:
                 print(f" > [{pdf_filename}] Rate limit hit (429) during JSON retrieval for job {job_id}.")
                 # Apply longer delay before next retry
                 wait_time = delay * (attempt + 2)
                 print(f" > [{pdf_filename}] Waiting {wait_time} seconds due to rate limit...")
                 time.sleep(wait_time)
                 continue # Skip normal delay, go to next attempt

            elif response.status_code >= 500:
                 print(f" > [{pdf_filename}] Server error ({response.status_code}) encountered for job {job_id}.")
                 # Server error, retry might help, fall through

            else:
                 print(f" > [{pdf_filename}] Unexpected client error during JSON retrieval for job {job_id}: {response.status_code} - {response.text}")
                 return None # Stop retries for unexpected client errors

            # If not successful and not stopped, wait before retrying
            if attempt < retries - 1:
                 wait_time = delay * (attempt + 1) # Basic exponential backoff
                 print(f" > [{pdf_filename}] Waiting {wait_time} seconds before next JSON retry for job {job_id}...")
                 time.sleep(wait_time)
            else:
                 print(f" > [{pdf_filename}] Max retries reached for JSON retrieval for job {job_id}.")
                 return None

        except requests.exceptions.RequestException as e:
            print(f" > [{pdf_filename}] Network or request error on JSON attempt {attempt + 1} for job {job_id}: {e}")
            if attempt < retries - 1:
                 wait_time = delay * (attempt + 1)
                 print(f" > [{pdf_filename}] Waiting {wait_time} seconds before retry...")
                 time.sleep(wait_time)
            else:
                 print(f" > [{pdf_filename}] Max retries reached for JSON after network error for job {job_id}.")
                 return None
    # Fallback if loop completes without returning (should theoretically not happen with current logic)
    return None

async def process_pdf(pdf_path, output_dir, semaphore):
    """
    Process a single PDF file with semaphore for concurrency control
    """
    async with semaphore:
        pdf_filename = os.path.basename(pdf_path)
        pdf_base_name_no_ext = os.path.splitext(pdf_filename)[0]
        output_file_json = os.path.join(output_dir, f"{pdf_base_name_no_ext}{OUTPUT_SUFFIX}.json")

        print(f"Processing Policies '{pdf_filename}'...")
        start_time_pdf = time.time()

        # Submit the job with both API keys
        job_id = submit_parsing_job_direct_api(
            pdf_path, LLAMA_API_KEY, VENDOR_API_KEY, VENDOR_MODEL
        )

        # --- Retrieve results if submission was successful ---
        if isinstance(job_id, str):  # Success: got a job ID (string)
            # Wait a bit before trying to fetch results
            initial_wait = 10
            print(f" > [{pdf_filename}] Waiting {initial_wait} seconds before first JSON retrieval attempt...")
            await asyncio.sleep(initial_wait)

            # Use the same Llama API key to retrieve results
            json_result = get_json_results(job_id, LLAMA_API_KEY, pdf_filename)

            if json_result:
                try:
                    # Add document type to result metadata
                    if isinstance(json_result, dict):
                        json_result["document_type"] = "Policies"
                    elif isinstance(json_result, list) and all(isinstance(item, dict) for item in json_result):
                        for item in json_result:
                            item["document_type"] = "Policies"
                    
                    with open(output_file_json, 'w', encoding='utf-8') as f:
                        json.dump(json_result, f, indent=2, ensure_ascii=False)
                    print(f" > [{pdf_filename}] JSON results successfully saved to: {output_file_json}")
                    return True, output_file_json
                except Exception as e:
                    print(f" > [{pdf_filename}] Error saving JSON results to file '{output_file_json}': {e}")
                    traceback.print_exc()
            else:
                print(f" > [{pdf_filename}] Failed to retrieve JSON results for job ID: {job_id}.")
        else:
            # Submission failed or skipped, message already printed
            pass

        end_time_pdf = time.time()
        print(f"Finished processing '{pdf_filename}'. Time elapsed: {end_time_pdf - start_time_pdf:.2f} seconds")
        return False, None

async def process_all_pdfs(source_dir, output_dir, concurrency_limit=None):
    """
    Process all PDF files in the specified directory that don't have corresponding JSON files
    
    Args:
        source_dir: Directory containing PDF files
        output_dir: Directory to save parsed results
        concurrency_limit: Optional limit on concurrent processing
        
    Returns:
        Tuple of (success_status, list_of_output_paths)
    """
    if concurrency_limit is None:
        concurrency_limit = CONCURRENCY_LIMIT
        
    # Get list of all PDF files in the directory
    pdf_files = glob.glob(os.path.join(source_dir, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return False, []
    
    print(f"Found {len(pdf_files)} PDF files in total")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for PDFs that don't have corresponding JSON files in the output_dir
    files_to_process = []
    for pdf_path in pdf_files:
        pdf_base_name_no_ext = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file_json = os.path.join(output_dir, f"{pdf_base_name_no_ext}{OUTPUT_SUFFIX}.json")
        
        # Skip if JSON already exists
        if os.path.exists(output_file_json):
            continue
            
        files_to_process.append(pdf_path)
    
    print(f"Found {len(files_to_process)} PDF files to process")
    
    if not files_to_process:
        print("Nothing to process. All PDFs already have corresponding JSON files.")
        return True, []
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    # Process files concurrently with semaphore control
    print(f"Starting processing with concurrency limit of {concurrency_limit}...")
    start_time = time.time()
    
    tasks = [process_pdf(pdf_path, output_dir, semaphore) for pdf_path in files_to_process]
    results = await asyncio.gather(*tasks)
    
    # Collect successful outputs
    success_count = 0
    output_paths = []
    
    for success, path in results:
        if success and path:
            success_count += 1
            output_paths.append(path)
    
    end_time = time.time()
    print(f"Processing complete. Successfully processed {success_count}/{len(files_to_process)} files.")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
    
    return success_count == len(files_to_process), output_paths

def save_json_result(json_result, filename, output_dir, job_id):
    """Save JSON result to a file and return success status and output path"""
    output_json_filename = os.path.splitext(filename)[0] + OUTPUT_SUFFIX + ".json"
    output_json_path = os.path.join(output_dir, output_json_filename)
    try:
        with open(output_json_path, "w", encoding="utf-8") as f_out:
            json.dump(json_result, f_out, indent=4, ensure_ascii=False)
        print(f"Successfully saved result to: {output_json_path} for job {job_id}")
        return True, output_json_path
    except Exception as e:
        print(f"Error saving JSON result for {filename} (Job ID: {job_id}): {e}")
        return False, None

def parse_document(pdf_path: str, output_dir: str = None) -> Tuple[bool, Optional[str]]:
    """
    Parse an Policies document using the direct API approach for single file processing.
    
    Args:
        pdf_path: Path to the PDF file to parse
        output_dir: Directory to save parsed results
        
    Returns:
        Tuple of (success status, output path or None)
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return False, None
        
    # Ensure output directory exists and is absolute
    if output_dir is None:
        output_dir = os.getcwd()
    elif not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename for logging
    filename = os.path.basename(pdf_path)
    pdf_base_name_no_ext = os.path.splitext(filename)[0]
    result_file_path = os.path.join(output_dir, f"{pdf_base_name_no_ext}{OUTPUT_SUFFIX}.json")
    
    # Check if result file already exists
    if os.path.exists(result_file_path):
        print(f"Skipping {filename} - Result file already exists at {result_file_path}")
        return True, result_file_path
        
    print(f"Parsing Policies document: {filename}")
    
    # Submit job using direct API approach
    job_id = submit_parsing_job_direct_api(
        pdf_path, LLAMA_API_KEY, VENDOR_API_KEY, VENDOR_MODEL
    )
    
    if not isinstance(job_id, str):
        print(f"Failed to submit parsing job for {filename}")
        return False, None
    
    # Save job info
    job_info = {"filename": filename, "job_id": job_id, "document_type": "Policies"}
    jobs_output_path = os.path.join(output_dir, "job_info.json")
    all_job_data = []
    
    if os.path.exists(jobs_output_path):
        try:
            with open(jobs_output_path, 'r', encoding='utf-8') as f:
                all_job_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_job_data = []
            
    all_job_data.append(job_info)
    
    try:
        with open(jobs_output_path, "w", encoding="utf-8") as f_json:
            json.dump(all_job_data, f_json, indent=4)
    except Exception as e:
        print(f"Error saving job information: {e}")
    
    # Try to retrieve results with multiple attempts
    # Wait a bit before first attempt
    print(f"Waiting 10 seconds before checking for results...")
    time.sleep(10)
    
    json_result = get_json_results(job_id, LLAMA_API_KEY, filename)
    
    if json_result:
        # Add document type to result
        if isinstance(json_result, dict):
            json_result["document_type"] = "Policies"
        elif isinstance(json_result, list) and all(isinstance(item, dict) for item in json_result):
            for item in json_result:
                item["document_type"] = "Policies"
                
        # Save successful result
        success, output_path = save_json_result(json_result, filename, output_dir, job_id)
        return success, output_path
    
    print(f"Could not retrieve results for job {job_id}.")
    return False, None

def parse_document_sync(pdf_path: str, output_dir: str = None) -> Tuple[bool, Optional[str]]:
    """
    Synchronous wrapper for parse_document.
    
    Args:
        pdf_path: Path to the PDF file to parse
        output_dir: Directory to save parsed results
        
    Returns:
        Tuple of (success status, output path or None)
    """
    return parse_document(pdf_path, output_dir)

# Example usage
if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Parse Policies documents using direct API approach")
    arg_parser.add_argument("--source-dir", help="Directory containing PDF files to parse", required=True)
    arg_parser.add_argument("--output-dir", help="Directory to save the parsed JSON results", required=True)
    arg_parser.add_argument("--concurrency", type=int, help="Number of concurrent files to process", default=CONCURRENCY_LIMIT)
    
    args = arg_parser.parse_args()

    source_directory = args.source_dir
    output_directory = args.output_dir
    concurrency = args.concurrency

    if not os.path.isdir(source_directory):
        print(f"Error: Source directory '{source_directory}' not found or is not a directory.")
        exit(1)
        
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Starting process for Policies PDFs in: {source_directory}")
    print(f"JSON results will be saved to: {output_directory}")
    print(f"Using concurrency limit of: {concurrency}")
    
    try:
        start_time = time.time()
        success, output_paths = asyncio.run(process_all_pdfs(source_directory, output_directory, concurrency))
        end_time = time.time()
        
        print(f"Overall process {'completed successfully' if success else 'completed with some failures'}.")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
        if output_paths:
            print(f"Successfully processed {len(output_paths)} files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() 