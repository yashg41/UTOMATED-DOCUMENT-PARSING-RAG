#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Document Parser for Multiple Document Types

This script processes PDFs for various document types and extracts metadata and content
for semantic search and information retrieval:
- Office Forms
- RA Database
- Forms and Checklists 
- Standard eForm
- Fleet Alert

It uses OpenAI's GPT-4o model to analyze document content and structure.
"""

import os
import base64
import io
import json
import time
import glob
import argparse
import concurrent.futures
from openai import OpenAI
from pdf2image import convert_from_path, pdfinfo_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Default values - these will be overridden by environment variables or command line args
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_INPUT_DIR = os.getenv("INPUT_DIRECTORY", "documents")
BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIRECTORY", "parsed_json_results")

# Document type configuration
DOCUMENT_TYPES = [
    "Office Forms",
    "RA Database",
    "Forms and Checklists",
    "Standard eForm",
    "Fleet Alert"
]

# Per-PDF processing settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "7"))  # Default concurrent processes
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Max pages per API call
OVERLAP = int(os.getenv("OVERLAP", "0"))  # Pages to overlap between batches
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "PNG")
IMAGE_DETAIL = os.getenv("IMAGE_DETAIL", "auto")
MAX_TOKENS_PER_BATCH = int(os.getenv("MAX_TOKENS_PER_BATCH", "3000"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- System prompts for different document types ---
BASE_SYSTEM_PROMPT = """
You are a document analysis agent responsible for processing {doc_type} documents. Your objective is to extract metadata and generate a semantically meaningful description that can be used in semantic or vector-based document search systems.

Your task involves two parts: metadata extraction and short-form description.

---

## 1. **Extract Metadata**

From the provided pages, prioritize the **first page**. Use subsequent pages only if required.

Extract the following fields:

- **"documentName"**: Identify the most prominent title of the document, typically found centered or bold at the top. If no reasonable candidate is found, return `null`.

- **"revDate"**: Search for a date labeled as "Revision Date", "Rev. Date", "Updated On", or similar. Normalize this to ISO 8601 format (`YYYY-MM-DD`). If only month/year is available, convert to the first day of that month. If absent, return `null`.

- **"RevNo"**: Identify any revision indicator such as "Rev. 2", "Revision 1.0", "Issue A", etc. Normalize to the string (e.g., `"Rev. 2"`). If not found, return `null`.

---

## 2. **Document description for Lookup**

Generate a short semantic description in the field **"originalText"**:

- Write a **2-3 sentence** description that describes the **type of document** and its **intended purpose**.
- Mention prominent section headers, field labels, and keywords found in the document.
- Highlight searchable terms related to the document's domain.
- Avoid interpretations — aim for semantic **recallability** and **search relevance**.
- Give a list of keywords relevant to the document.
---

## Output Format

You must return a valid JSON object with **exactly** the following keys:
{
"documentName": "string or null",
"revDate": "YYYY-MM-DD or null",
"RevNo": "string or null",
"originalText": "short descriptive paragraph with keywords"
}

- Return `null` (the literal JSON null) if any field is missing.
- Do **not** add extra text or explanations.
- Do **not** format as Markdown or wrap in code blocks.
- Output must begin with `{` and end with `}`.
"""

# Custom prompts for each document type
DOCUMENT_TYPE_PROMPTS = {
    "Office Forms": BASE_SYSTEM_PROMPT.replace("{doc_type}", "Office Forms") + """
For Office Forms, focus on:
- Administrative purpose of the form
- Department/team it belongs to
- Type of information collected
- Any approval workflows indicated
""",

    "RA Database": BASE_SYSTEM_PROMPT.replace("{doc_type}", "RA Database") + """
For RA (Risk Assessment) Database documents, focus on:
- Risk categories being assessed
- Hazard identification terminology
- Control measures mentioned
- Risk matrices or scoring systems
""",

    "Forms and Checklists": BASE_SYSTEM_PROMPT.replace("{doc_type}", "Forms and Checklists") + """
For Forms and Checklists, focus on:
- Process or operation being checked
- Compliance requirements referenced
- Inspection items or checkpoints
- Frequency of checks if mentioned
""",

    "Standard eForm": BASE_SYSTEM_PROMPT.replace("{doc_type}", "Standard eForm") + """
For Standard eForms, focus on:
- Electronic submission workflows
- Data fields and required information
- Integration points with other systems
- Digital approval or signature requirements
""",

    "Fleet Alert": BASE_SYSTEM_PROMPT.replace("{doc_type}", "Fleet Alert") + """
For Fleet Alerts, focus on:
- Type of alert (safety, maintenance, operational)
- Vessel types affected
- Action items or requirements
- Timeline or urgency indicators
- Technical specifications if relevant
"""
}

# --- Helper Function to Call OpenAI API ---
def call_openai_api(client, model, messages, response_format_type="json_object", max_tokens=1500):
    """Makes an API call and handles potential errors."""
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": response_format_type} if response_format_type else None,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except TypeError as e:
        # Handle the "got an unexpected keyword argument 'proxies'" error
        if "unexpected keyword argument 'proxies'" in str(e):
            print("Detected proxies error, attempting to create a new client without proxies")
            try:
                # Create a new client instance without any proxy
                new_client = OpenAI(api_key=OPENAI_API_KEY)
                # Try again with the new client
                response = new_client.chat.completions.create(
                    model=model,
                    response_format={"type": response_format_type} if response_format_type else None,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as retry_e:
                print(f"Error during retry OpenAI API call: {retry_e}")
                return None
    except Exception as e:
        # Provide more context about the error if possible
        print(f"Error during OpenAI API call: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"API Response Status: {e.response.status_code}")
            try:
                print(f"API Response Body: {e.response.json()}")
            except ValueError:
                print(f"API Response Body: {e.response.text}")
        return None

# --- Helper Function to Convert Page to Image Data URI ---
def get_image_data_uri(pdf_path, page_num_zero_based, img_format='PNG'):
    """Converts a single PDF page to a base64 Data URI."""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num_zero_based + 1,
            last_page=page_num_zero_based + 1
        )
        if not images:
            print(f"Warning: Could not convert page {page_num_zero_based + 1} of {pdf_path}.")
            return None
        image = images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format=img_format)
        byte_data = byte_stream.getvalue()
        base64_string = base64.b64encode(byte_data).decode('utf-8')
        mime_type = f"image/{img_format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"

    except Exception as e:
        print(f"Error converting page {page_num_zero_based + 1} of {pdf_path}: {e}")
        return None

# --- Process Single PDF File ---
def process_pdf(pdf_file_path, doc_type, output_dir):
    """Process a single PDF file and return results."""
    pdf_start_time = time.time()
    pdf_basename = os.path.basename(pdf_file_path)
    pdf_base_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
    safe_base_name = "".join(c for c in pdf_base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    
    # Use the output_dir parameter passed to the function, not the global BASE_OUTPUT_DIR
    doc_type_output_dir = output_dir
    os.makedirs(doc_type_output_dir, exist_ok=True)
    
    # Check if result already exists in the output directory
    result_filename = f"{safe_base_name}_result.json"
    result_path = os.path.join(doc_type_output_dir, result_filename)
    if os.path.exists(result_path):
        print(f"\n>>> Skipping {pdf_basename} as result file already exists: {result_filename}")
        return {
            "status": "skipped",
            "file": pdf_basename,
            "doc_type": doc_type,
            "message": "Result file already exists"
        }
    
    print(f"\n>>> Starting to process [{doc_type}]: {pdf_basename}")
    
    # Debug proxy settings
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
    
    # Create a client for this process - directly import and initialize with no proxies
    try:
        # Import httpx directly to create a clean client with no proxy settings
        import httpx
        
        # Create a fresh httpx client with no proxy configuration
        http_client = httpx.Client()
        
        # Initialize OpenAI client with our custom http_client
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=http_client  # Use our clean httpx client
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return {"status": "error", "file": pdf_basename, "doc_type": doc_type, "message": f"Error initializing OpenAI client: {e}"}
    
    # Ensure output directory exists
    os.makedirs(doc_type_output_dir, exist_ok=True)

    # Get page count
    try:
        pdf_info = pdfinfo_from_path(pdf_file_path)
        page_count = pdf_info.get('Pages', 0)
        if page_count == 0:
            print(f"Error: Could not determine page count for '{pdf_basename}' or PDF is empty.")
            return {"status": "error", "file": pdf_basename, "doc_type": doc_type, "message": "Could not determine page count or PDF is empty"}
        
        print(f"PDF '{pdf_basename}' has {page_count} pages.")
    except Exception as e:
        print(f"Error getting PDF info for '{pdf_basename}': {e}")
        return {"status": "error", "file": pdf_basename, "doc_type": doc_type, "message": f"Error getting PDF info: {e}"}

    # Calculate batch ranges
    batches = []
    if page_count <= BATCH_SIZE:
        batches.append((0, page_count - 1))
    else:
        step = BATCH_SIZE - OVERLAP
        for i in range(0, page_count, step):
            start_page = i
            end_page = min(i + BATCH_SIZE - 1, page_count - 1)
            batches.append((start_page, end_page))
            if end_page == page_count - 1:
                break

    print(f"Processing '{pdf_basename}' in {len(batches)} batches: {[(s+1, e+1) for s, e in batches]}")

    processed_batches = 0
    batch_results = []
    
    # Get the appropriate system prompt for this document type
    system_prompt = DOCUMENT_TYPE_PROMPTS.get(doc_type, BASE_SYSTEM_PROMPT.replace("{doc_type}", doc_type))

    # Process each batch
    for i, (start_page, end_page) in enumerate(batches):
        batch_num = i + 1
        pages_in_batch = list(range(start_page, end_page + 1))
        user_friendly_page_range = f"{start_page + 1}_to_{end_page + 1}"
        print(f"--- Processing Batch {batch_num}/{len(batches)} (Pages {start_page + 1}-{end_page + 1}) of '{pdf_basename}' ---")

        # Convert pages to images
        batch_image_data_uris = []
        for page_num in pages_in_batch:
            data_uri = get_image_data_uri(pdf_file_path, page_num, IMAGE_FORMAT)
            if data_uri:
                batch_image_data_uris.append(data_uri)
            else:
                print(f"Skipping page {page_num + 1} of '{pdf_basename}' due to conversion error.")

        if not batch_image_data_uris:
            print(f"Skipping Batch {batch_num} of '{pdf_basename}' as no pages could be converted to images.")
            continue

        # Construct API payload
        user_message_content = [
            {
                "type": "text",
                "text": f"Analyze the content of these {len(batch_image_data_uris)} document pages (representing pages {start_page + 1} through {end_page + 1} of the original document) according to the system instructions. Provide a single JSON output summarizing this batch."
            }
        ]
        for data_uri in batch_image_data_uris:
            user_message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_uri,
                    "detail": IMAGE_DETAIL
                }
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        # Call OpenAI API
        print(f"Sending batch {batch_num} analysis request to OpenAI API for '{pdf_basename}'...")
        batch_analysis_content = call_openai_api(
            client, MODEL, messages, "json_object", MAX_TOKENS_PER_BATCH
        )

        # Determine output path for this batch
        if len(batches) > 1:
            pdf_subfolder = os.path.join(doc_type_output_dir, safe_base_name)
            os.makedirs(pdf_subfolder, exist_ok=True)
            output_path = pdf_subfolder
        else:
            output_path = doc_type_output_dir

        output_json_path = os.path.join(output_path, f"{safe_base_name}_batch_{batch_num}_pages_{user_friendly_page_range}_output.json")

        # Save the response
        if batch_analysis_content:
            try:
                parsed_json = json.loads(batch_analysis_content)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, indent=4, ensure_ascii=False)
                print(f"Successfully saved JSON for batch {batch_num} of '{pdf_basename}'")
                processed_batches += 1
                batch_results.append({
                    "batch": batch_num,
                    "status": "success",
                    "output_path": output_json_path
                })
            except json.JSONDecodeError:
                print(f"Warning: OpenAI response for batch {batch_num} of '{pdf_basename}' was not valid JSON.")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    f.write(batch_analysis_content)
                batch_results.append({
                    "batch": batch_num,
                    "status": "warning",
                    "message": "Response was not valid JSON",
                    "output_path": output_json_path
                })
            except Exception as e:
                print(f"Error saving file '{output_json_path}': {e}")
                batch_results.append({
                    "batch": batch_num,
                    "status": "error",
                    "message": f"Error saving file: {e}"
                })
        else:
            print(f"Warning: No valid response received from API for batch {batch_num} of '{pdf_basename}'.")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                f.write(str(batch_analysis_content) if batch_analysis_content is not None else "None")
            batch_results.append({
                "batch": batch_num,
                "status": "error",
                "message": "No valid response received from API",
                "output_path": output_json_path
            })

        # Add a small delay between batch requests
        if batch_num < len(batches):
            time.sleep(1)  # shorter delay since we're already parallelizing files

    pdf_end_time = time.time()
    pdf_time_taken = pdf_end_time - pdf_start_time
    
    # Process the results if multiple batches
    if len(batches) > 1 and processed_batches > 0:
        combine_batches(safe_base_name, doc_type_output_dir)
    
    print(f">>> Completed processing '{pdf_basename}'. Processed {processed_batches} of {len(batches)} batches in {pdf_time_taken:.2f} seconds.")
    
    return {
        "status": "success" if processed_batches > 0 else "error",
        "file": pdf_basename,
        "doc_type": doc_type,
        "batches_processed": processed_batches,
        "total_batches": len(batches),
        "time_taken": pdf_time_taken,
        "results": batch_results
    }

# --- Combine Batch Results for a Single PDF ---
def combine_batches(pdf_name, output_dir):
    """Combine multiple batch results for a single PDF."""
    pdf_subfolder = os.path.join(output_dir, pdf_name)
    
    if not os.path.exists(pdf_subfolder) or not os.path.isdir(pdf_subfolder):
        print(f"No subfolder found for {pdf_name}, skipping combination.")
        return
    
    # Get all JSON files in the subfolder
    json_files = glob.glob(os.path.join(pdf_subfolder, "*.json"))
    if not json_files:
        print(f"No JSON files found in {pdf_name} subfolder. Skipping...")
        return
    
    # Sort files to ensure consistent ordering
    json_files.sort()
    
    # Initialize variables for metadata and combined analysis
    metadata = {}
    combined_original_text = ""
    
    # Process each JSON file
    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # If this is the first file, save the metadata
            if i == 0:
                metadata = {
                    "documentName": data.get("documentName"),
                    "revDate": data.get("revDate"),
                    "RevNo": data.get("RevNo")
                }
            
            # Extract originalText and append to combined string
            original_text = data.get("originalText", "")
            if original_text:
                if combined_original_text:
                    combined_original_text += "\n\n" + original_text
                else:
                    combined_original_text = original_text
        except Exception as e:
            print(f"Error processing {os.path.basename(json_file)}: {str(e)}")
    
    # Create combined JSON
    combined_json = metadata.copy()
    combined_json["originalText"] = combined_original_text
    
    # Save combined JSON to main directory with _result suffix
    output_filename = f"{pdf_name}_result.json"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(combined_json, f, indent=4)
        print(f"Successfully saved combined JSON to: {output_path}")
    except Exception as e:
        print(f"Error saving combined JSON for {pdf_name}: {str(e)}")

def process_standalone_jsons(output_dir):
    """Process any standalone JSON files (single batch PDFs)."""
    standalone_jsons = glob.glob(os.path.join(output_dir, "*_batch_*_output.json"))
    if not standalone_jsons:
        return
        
    print(f"\nProcessing {len(standalone_jsons)} standalone JSON files...")
    for json_file in standalone_jsons:
        file_name = os.path.basename(json_file)
        
        # Skip files already ending with _result.json
        if "_result.json" in file_name:
            continue
        
        # Extract PDF name from the file name
        match = os.path.splitext(file_name)[0].split("_batch_")[0]
        if match:
            output_filename = f"{match}_result.json"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if result already exists
            if os.path.exists(output_path):
                print(f"Skipping {file_name} as result already exists: {output_filename}")
                continue
                
            try:
                # Read the original file
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Write to the new file
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Renamed {file_name} to {output_filename}")
                
                # Remove the original file
                os.remove(json_file)
            except Exception as e:
                print(f"Error processing standalone file {file_name}: {str(e)}")

def process_document_type(doc_type, input_dir=None, output_dir=None, openai_key=None, max_workers=None):
    """
    Process all PDFs for a given document type
    
    Args:
        doc_type: Document type to process
        input_dir: Optional override for input directory
        output_dir: Optional override for output directory
        openai_key: Optional override for OpenAI API key
        max_workers: Optional override for max workers
        
    Returns:
        Result dictionary
    """
    print(f"\n======= Processing Document Type: {doc_type} =======\n")
    
    # Update configuration if needed
    global OPENAI_API_KEY, MAX_WORKERS
    
    if openai_key:
        OPENAI_API_KEY = openai_key
    if max_workers:
        MAX_WORKERS = max_workers
        
    # Set input directory for this document type
    if input_dir:
        input_doc_dir = os.path.join(input_dir, doc_type)
    else:
        input_doc_dir = os.path.join(BASE_INPUT_DIR, doc_type)
        
    # Set output directory for this document type
    if output_dir:
        output_doc_dir = os.path.join(output_dir, doc_type)
    else:
        output_doc_dir = os.path.join(BASE_OUTPUT_DIR, doc_type)
    
    # Ensure directories exist
    if not os.path.exists(input_doc_dir):
        print(f"Notice: Input directory '{input_doc_dir}' does not exist. No files to process for {doc_type}.")
        return {
            "doc_type": doc_type,
            "status": "success",  # Changed from "error" to "success"
            "message": "Input directory does not exist - no files to process",
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }
    
    os.makedirs(output_doc_dir, exist_ok=True)
    
    # Get all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_doc_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_doc_dir}")
        return {
            "doc_type": doc_type,
            "status": "success",
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }
    
    print(f"Found {len(pdf_files)} PDF files to process in {input_doc_dir}")
    
    # Before submitting PDFs for processing, filter out those that already have results
    pdf_files_to_process = []
    skipped_files = []

    for pdf in pdf_files:
        pdf_base_name = os.path.splitext(os.path.basename(pdf))[0]
        safe_base_name = "".join(c for c in pdf_base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        result_path = os.path.join(output_doc_dir, f"{safe_base_name}_result.json")
        
        if os.path.exists(result_path):
            skipped_files.append(pdf)
            print(f"  ↷ Skipping {os.path.basename(pdf)} - result file already exists")
        else:
            pdf_files_to_process.append(pdf)
            print(f"  - {os.path.basename(pdf)}")

    print(f"Files to process: {len(pdf_files_to_process)}, Files skipped: {len(skipped_files)}")
    
    if not pdf_files_to_process:
        print("No files to process. All files already have results.")
        return {
            "doc_type": doc_type,
            "status": "success",
            "processed": 0,
            "skipped": len(skipped_files),
            "failed": 0
        }
    
    # Process PDFs in parallel using ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {executor.submit(process_pdf, pdf, doc_type, output_doc_dir): pdf for pdf in pdf_files_to_process}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                pdf_basename = os.path.basename(pdf)
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"✓ Successfully processed {pdf_basename}")
                elif status == "skipped":
                    print(f"⟳ Skipped {pdf_basename}: {result.get('message', 'Unknown reason')}")
                else:
                    print(f"✗ Failed to process {pdf_basename}: {result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing {os.path.basename(pdf)}: {e}")
                results.append({
                    "status": "error",
                    "file": os.path.basename(pdf),
                    "doc_type": doc_type,
                    "message": str(e)
                })
    
    # Process any standalone JSON files
    process_standalone_jsons(output_doc_dir)
    
    # Calculate statistics
    successful = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = len(pdf_files_to_process) - successful - skipped
    
    return {
        "doc_type": doc_type,
        "status": "success",
        "processed": successful,
        "skipped": len(skipped_files) + skipped,
        "failed": failed
    }

def update_config_from_args(args):
    """Update configuration from command line arguments."""
    # Use a function to update global variables instead of global statement
    global OPENAI_API_KEY, BASE_INPUT_DIR, BASE_OUTPUT_DIR, MAX_WORKERS
    
    if args.max_workers:
        MAX_WORKERS = args.max_workers
    
    if args.api_key:
        OPENAI_API_KEY = args.api_key
    
    if args.input_dir:
        BASE_INPUT_DIR = args.input_dir
    
    if args.output_dir:
        BASE_OUTPUT_DIR = args.output_dir

def main():
    """Main function to coordinate document processing."""
    overall_start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse different document types using OpenAI")
    parser.add_argument("--doc-type", choices=DOCUMENT_TYPES + ["all"],
                        default="all", help="Document type to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help=f"Maximum number of concurrent processes (default: {MAX_WORKERS})")
    parser.add_argument("--api-key", help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--input-dir", help="Base input directory (overrides default)")
    parser.add_argument("--output-dir", help="Base output directory (overrides default)")
    
    args = parser.parse_args()
    
    # Update configuration from arguments
    update_config_from_args(args)
    
    # Check API key
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not set. Please set OPENAI_API_KEY environment variable or use --api-key.")
        return 1
    
    print(f"=== General Document Parser ===")
    print(f"Input directory: {BASE_INPUT_DIR}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"Concurrent workers: {MAX_WORKERS}")
    print(f"OpenAI model: {MODEL}")
    
    # Determine which document types to process
    doc_types_to_process = DOCUMENT_TYPES if args.doc_type == "all" else [args.doc_type]
    
    # Process each document type
    summary_results = []
    for doc_type in doc_types_to_process:
        type_start_time = time.time()
        result = process_document_type(doc_type, args.input_dir, args.output_dir, args.api_key, args.max_workers)
        type_end_time = time.time()
        
        result["time_taken"] = type_end_time - type_start_time
        summary_results.append(result)
    
    # Print overall summary
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    print("\n=== Processing Summary ===")
    print(f"Total document types processed: {len(doc_types_to_process)}")
    
    total_processed = sum(r.get("processed", 0) for r in summary_results)
    total_skipped = sum(r.get("skipped", 0) for r in summary_results)
    total_failed = sum(r.get("failed", 0) for r in summary_results)
    total_files = total_processed + total_skipped + total_failed
    
    print(f"Total files processed: {total_processed}")
    print(f"Total files skipped: {total_skipped}")
    print(f"Total files failed: {total_failed}")
    
    if total_files > 0:
        print(f"Success rate: {(total_processed / (total_processed + total_failed)) * 100:.1f}%")
    
    print(f"Total time taken: {total_time:.2f} seconds")
    
    # Print breakdown by document type
    print("\nBreakdown by Document Type:")
    for result in summary_results:
        print(f"  {result['doc_type']}: {result['processed']} processed, {result['skipped']} skipped, "
              f"{result['failed']} failed ({result['time_taken']:.2f} seconds)")
    
    return 0

if __name__ == "__main__":
    exit(main()) 