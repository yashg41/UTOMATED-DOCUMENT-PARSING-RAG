#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document Automation Main Script

This script orchestrates the entire document automation workflow:
1. Scraping documents from a website
2. Downloading files to organized folders
3. Converting non-PDF files to PDF format
4. Parsing documents to JSON:
   - ISM Manual documents using LlamaIndex
   - Various document types (Office Forms, RA Database, etc.) using OpenAI
5. Post-processing parsed JSON files to extract page and md content
6. Processing JSON with LLM to extract structured data
7. Final post-processing to flatten and structure JSON for ISM Manual and Office Manual
8. Upload documents to MongoDB and add MongoDB IDs to JSON files
"""

import os
import sys
import argparse
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import glob
import json
import re
import csv
from pathlib import Path
import time
import subprocess
import shutil  # Add import for shutil

# Import custom modules
import document_scrap
import document_to_pdf
from document_parser_ism import parse_document_sync as parse_ism_document
from document_parser_manual import parse_document_sync as parse_office_manual
import document_parser_general
import llamaparse_processing
import post_processing
import document_s3_uploader # Import the document_s3_uploader module
import upload_to_mongodb  # Import the MongoDB upload module
import mongodb_to_typesense  # Import the new module for Typesense sync

# Load environment variables from .env file
load_dotenv()

# Create a timestamped log file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"document_automation_{timestamp}.log"

# Setup logging with timestamped file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create a class to capture stdout and stderr
class LogCapture:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set up stdout and stderr redirection to log file
sys.stdout = LogCapture(log_filename)
sys.stderr = LogCapture(log_filename)

# Setup exception logging
def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set the exception hook
sys.excepthook = log_uncaught_exceptions

def parse_arguments():
    """Parse command line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(description="Document Automation Tool")
    
    # Add arguments with default values from environment variables
    parser.add_argument(
        "--username", 
        default=os.getenv("SCRAPER_USERNAME"),
        help="Username for website login (env: SCRAPER_USERNAME)"
    )
    parser.add_argument(
        "--password", 
        default=os.getenv("SCRAPER_PASSWORD"),
        help="Password for website login (env: SCRAPER_PASSWORD)"
    )
    parser.add_argument(
        "--download-username",
        default=os.getenv("DOWNLOADER_USERNAME"),
        help="Username for file downloads (env: DOWNLOADER_USERNAME)"
    )
    parser.add_argument(
        "--download-password",
        default=os.getenv("DOWNLOADER_PASSWORD"),
        help="Password for file downloads (env: DOWNLOADER_PASSWORD)"
    )
    parser.add_argument(
        "--adobe-id", 
        default=os.getenv("ADOBE_CLIENT_ID"),
        help="Adobe API client ID (env: ADOBE_CLIENT_ID)"
    )
    parser.add_argument(
        "--adobe-secret", 
        default=os.getenv("ADOBE_CLIENT_SECRET"),
        help="Adobe API client secret (env: ADOBE_CLIENT_SECRET)"
    )
    parser.add_argument(
        "--output-dir", 
        default=os.getenv("OUTPUT_DIRECTORY", "downloaded_data"),
        help="Directory to save downloaded files (env: OUTPUT_DIRECTORY, default: downloaded_data)"
    )
    parser.add_argument(
        "--json-output-dir",
        default=os.getenv("JSON_OUTPUT_DIR", "parsed_json_results"),
        help="Directory to save parsed JSON results (env: JSON_OUTPUT_DIR, default: parsed_json_results)"
    )
    parser.add_argument(
        "--only-scrape", 
        action="store_true", 
        help="Only scrape documents, don't convert to PDF"
    )
    parser.add_argument(
        "--only-convert", 
        action="store_true", 
        help="Only convert existing documents to PDF"
    )
    parser.add_argument(
        "--only-parse",
        action="store_true",
        help="Only parse documents to JSON"
    )
    parser.add_argument(
        "--parse-ism-only",
        action="store_true",
        help="Only parse ISM Manual documents to JSON"
    )
    parser.add_argument(
        "--parse-doc-type",
        choices=document_parser_general.DOCUMENT_TYPES + ["all", "ISM Manual", "Office Manual"],
        default=None,
        help="Parse specific document type(s) to JSON using OpenAI or LlamaIndex"
    )
    parser.add_argument(
        "--only-add-chunks",
        action="store_true",
        help="Only add chunk numbers to existing JSON files"
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API Key for document parsing (env: OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=int(os.getenv("MAX_WORKERS", "3")),
        help="Maximum number of concurrent PDF conversions/parsing operations (env: MAX_WORKERS, default: 3)"
    )
    parser.add_argument(
        "--domain", 
        default=os.getenv("SCRAPER_DOMAIN", "docMap"),
        help="Domain for authentication (env: SCRAPER_DOMAIN, default: docMap)"
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """Validate command line arguments."""
    
    # Check for configuration conflicts
    # Modified logic to allow using --only-parse with --parse-doc-type together
    exclusive_modes = []
    
    if args.only_scrape:
        exclusive_modes.append("--only-scrape")
    if args.only_convert:
        exclusive_modes.append("--only-convert")
    if args.parse_ism_only:
        exclusive_modes.append("--parse-ism-only")
    if args.only_add_chunks:
        exclusive_modes.append("--only-add-chunks")
    
    # --only-parse and --parse-doc-type can be used together
    if args.only_parse and args.parse_doc_type is None:
        exclusive_modes.append("--only-parse")
    elif args.parse_doc_type is not None and not args.only_parse:
        exclusive_modes.append("--parse-doc-type")
    
    if len(exclusive_modes) > 1:
        logger.error("Cannot use more than one of --only-scrape, --only-convert, --only-parse, --parse-ism-only, --only-add-chunks, or --parse-doc-type at the same time.")
        return False
    
    # Check required arguments for scraping
    if not any([args.only_convert, args.only_parse, args.parse_ism_only, args.parse_doc_type, args.only_add_chunks]) and (not args.username or not args.password):
        logger.error("Username and password are required for document scraping.")
        logger.error("Set them using --username and --password or in the .env file as SCRAPER_USERNAME and SCRAPER_PASSWORD.")
        return False
    
    # Check required arguments for PDF conversion
    if not any([args.only_scrape, args.only_parse, args.parse_ism_only, args.parse_doc_type, args.only_add_chunks]) and (not args.adobe_id or not args.adobe_secret):
        logger.error("Adobe API client ID and secret are required for PDF conversion.")
        logger.error("Set them using --adobe-id and --adobe-secret or in the .env file as ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET.")
        return False
    
    # Check required arguments for ISM Manual parsing
    if ((args.only_parse and args.parse_doc_type is None) or args.parse_ism_only or 
        (args.parse_doc_type == "ISM Manual" or args.parse_doc_type == "all")) and not os.getenv("LLAMA_API_KEY"):
        logger.error("LLAMA_API_KEY environment variable is required for ISM document parsing.")
        logger.error("Set it in the .env file.")
        return False
    
    # Check required arguments for general document parsing
    if args.parse_doc_type is not None and not args.openai_api_key:
        logger.error("OpenAI API key is required for document type parsing.")
        logger.error("Set it using --openai-api-key or in the .env file as OPENAI_API_KEY.")
        return False
    
    return True

def parse_manual_documents(input_dir, output_dir, document_type):
    """
    Parse ISM Manual or Office Manual documents.
    
    Args:
        input_dir: Directory containing document PDFs
        output_dir: Directory to save parsed results
        document_type: "ISM Manual" or "Office Manual" or "Policies" or "Sample Document Type"
        
    Returns:
        dict: Result with success status and counts
    """
    logger.info(f"==== Starting {document_type} Document Parsing ====")
    
    # Determine the appropriate parser module based on document type
    if document_type == "ISM Manual":
        try:
            import document_parser_ism as parser_module
            module_name = "ISM Manual"
        except ImportError:
            logger.error("Failed to import document_parser_ism module")
            return {"success": False, "error": "Failed to import document_parser_ism module"}
    elif document_type == "Office Manual":
        try:
            import document_parser_manual as parser_module
            module_name = "Office Manual"
        except ImportError:
            logger.error("Failed to import document_parser_manual module")
            return {"success": False, "error": "Failed to import document_parser_manual module"}
    elif document_type == "Policies":
        try:
            import document_parser_policy as parser_module
            module_name = "Policies"
        except ImportError:
            logger.error("Failed to import document_parser_policy module")
            return {"success": False, "error": "Failed to import document_parser_policy module"}
    elif document_type == "Sample Document Type":
        try:
            import document_parser_Sample_Document_Type as parser_module
            module_name = "Sample Document Type"
        except ImportError:
            logger.error("Failed to import document_parser_Sample_Document_Type module")
            return {"success": False, "error": "Failed to import document_parser_Sample_Document_Type module"}
    else:
        logger.error(f"Unsupported document type: {document_type}")
        return {"success": False, "error": f"Unsupported document type: {document_type}"}
    
    # Find the directory for this document type
    document_dir = get_document_type_dir(input_dir, document_type, create_if_missing=False)
    if not document_dir:
        logger.warning(f"No directory found for {document_type} in {input_dir}")
        return {
            "success": True,
            "message": f"No directory found for {document_type} in {input_dir}",
            "processed_count": 0,
            "success_count": 0,
            "skipped_count": 0,
            "failed_count": 0
        }
    
    # Create output directory for this document type
    os.makedirs(output_dir, exist_ok=True)
    json_output_dir = os.path.join(output_dir, document_type)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Get all PDF files in the document type directory (not recursive)
    pdf_files = glob.glob(os.path.join(document_dir, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {document_dir}")
    
    # Check if we have files to process
    if not pdf_files:
        logger.info(f"No PDF files found in {document_dir}")
        return {
            "success": True,
            "message": f"No PDF files found in {document_dir}",
            "processed_count": 0,
            "success_count": 0,
            "skipped_count": 0,
            "failed_count": 0
        }
    
    # Process files with the parser module
    try:
        # Run the parser's process_all_pdfs function
        success, output_files = asyncio.run(parser_module.process_all_pdfs(
            document_dir, json_output_dir, concurrency_limit=3
        ))
        
        # Return the results
        return {
            "success": success,
            "processed_count": len(pdf_files),
            "success_count": len(output_files),
            "skipped_count": len(pdf_files) - len(output_files),
            "failed_count": len(pdf_files) - len(output_files),
            "output_files": output_files
        }
    except Exception as e:
        logger.exception(f"Error processing {document_type} PDFs: {e}")
        return {
            "success": False,
            "error": f"Error processing {document_type} PDFs: {str(e)}",
            "processed_count": len(pdf_files),
            "success_count": 0,
            "skipped_count": 0,
            "failed_count": len(pdf_files)
        }

def parse_documents_by_type(input_dir, output_dir, doc_type, max_workers, openai_api_key):
    """
    Parse documents by type using the document_parser_general module.
    
    Args:
        input_dir: Base directory containing document type folders
        output_dir: Base directory to save parsed JSON results
        doc_type: Document type to parse or "all" for all types
        max_workers: Maximum number of concurrent workers
        openai_api_key: OpenAI API key
        
    Returns:
        dict: Result summary with success and failure counts
    """
    logger.info(f"==== Starting Document Parsing for {doc_type if doc_type != 'all' else 'All'} Document Types ====")
    
    # Set environment variables for document_parser_general
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["INPUT_DIRECTORY"] = input_dir
    os.environ["OUTPUT_DIRECTORY"] = output_dir
    os.environ["MAX_WORKERS"] = str(max_workers)
    
    # Process the document types
    try:
        # Process specific document type or all types
        if doc_type == "all":
            doc_types_to_process = document_parser_general.DOCUMENT_TYPES
            logger.info(f"Processing all document types: {', '.join(doc_types_to_process)}")
        else:
            doc_types_to_process = [doc_type]
            logger.info(f"Processing document type: {doc_type}")
        
        # Process each document type
        summary_results = []
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        for dt in doc_types_to_process:
            logger.info(f"Starting processing for {dt}")
            
            # Call the document_parser_general function
            result = document_parser_general.process_document_type(
                dt, 
                input_dir=input_dir, 
                output_dir=output_dir, 
                openai_key=openai_api_key, 
                max_workers=max_workers
            )
            
            if result["status"] == "success":
                processed = result.get("processed", 0)
                skipped = result.get("skipped", 0)
                
                # Check if this was a missing directory case
                if "message" in result and "no files to process" in result["message"].lower():
                    logger.info(f"No files to process for {dt}: {result['message']}")
                elif processed > 0:
                    logger.info(f"Successfully processed {processed} files for {dt}")
                    
                if skipped > 0:
                    logger.info(f"Skipped {skipped} already processed files for {dt}")
                    
                total_processed += processed
                total_skipped += skipped
            else:
                logger.error(f"Error processing {dt}: {result.get('error', 'Unknown error')}")
            
            total_failed += result.get("failed", 0)
            summary_results.append(result)
            
        # Return overall results
        return {
            "success": True,  # Changed to always be true if no fatal errors occurred
            "processed": total_processed,
            "skipped": total_skipped,
            "success_count": total_processed,
            "failed_count": total_failed,
            "summary": summary_results
        }
    
    except Exception as e:
        logger.exception(f"Error during document type parsing: {e}")
        return {
            "success": False,
            "error": str(e),
            "processed": 0,
            "skipped": 0,
            "success_count": 0,
            "failed_count": 0
        }

def get_document_type_dir(base_dir, document_type, create_if_missing=True):
    """
    Find the directory for a specific document type within the base directory.
    Creates the directory if it doesn't exist and create_if_missing is True.
    
    Args:
        base_dir: Base directory to search in
        document_type: Document type to find the directory for
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        str or None: Path to the document type directory or None if not found
    """
    logger.info(f"Looking for directory for document type: {document_type} in {base_dir}")
    
    # Check if the base directory exists
    if not os.path.isdir(base_dir):
        logger.warning(f"Base directory not found: {base_dir}")
        if create_if_missing:
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created base directory: {base_dir}")
        else:
            return None
    
    # Look for direct match or specific directory structure
    for dirname in os.listdir(base_dir):
        # Skip non-directories
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.isdir(dir_path):
            continue
            
        dirname_lower = dirname.lower()
        if document_type == "ISM Manual":
            if dirname_lower == "ism manual" or dirname_lower == "ism" or (
                "ism" in dirname_lower and "manual" in dirname_lower and "office" not in dirname_lower
            ):
                logger.info(f"Found ISM Manual directory: {dir_path}")
                return dir_path
        elif document_type == "Office Manual":
            if dirname_lower == "office manual" or (
                "office" in dirname_lower and "manual" in dirname_lower
            ):
                logger.info(f"Found Office Manual directory: {dir_path}")
                return dir_path
        elif document_type == "Policies":
            if dirname_lower == "policies" or "policies" in dirname_lower:
                logger.info(f"Found Policies directory: {dir_path}")
                return dir_path
        elif document_type == "Sample Document Type":
            if dirname_lower == "sample document type" or dirname_lower == "sample_document_type" or (
                "sample" in dirname_lower and "document" in dirname_lower and "type" in dirname_lower
            ):
                logger.info(f"Found Sample Document Type directory: {dir_path}")
                return dir_path
        elif document_type in document_parser_general.DOCUMENT_TYPES:
            # For general document types, look for exact name or simplified version
            simplified_name = ''.join(c for c in document_type if c.isalnum()).lower()
            dirname_simplified = ''.join(c for c in dirname if c.isalnum()).lower()
            if dirname_lower == document_type.lower() or simplified_name in dirname_simplified:
                logger.info(f"Found {document_type} directory: {dir_path}")
                return dir_path
    
    # If no matching directory found, create only if create_if_missing is True
    if create_if_missing:
        new_dir = os.path.join(base_dir, document_type)
        os.makedirs(new_dir, exist_ok=True)
        logger.info(f"Created directory for {document_type} at: {new_dir}")
        return new_dir
    else:
        logger.info(f"No directory found for {document_type} and not creating one")
        return None

def process_manual_json_files(json_output_dir):
    """
    Process parsed JSON files for ISM and Office Manual documents to extract page and md content.
    
    Args:
        json_output_dir: Base directory containing the parsed JSON results
        
    Returns:
        dict: Result summary with success and failure counts
    """
    logger.info("==== Post-processing parsed JSON files for ISM and Office Manual ====")
    
    subfolders = ['ISM Manual', 'OFFICE MANUAL','Policies','Sample Document Type']
    all_files_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Process each specified subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(json_output_dir, subfolder)
        
        if not os.path.exists(subfolder_path):
            logger.warning(f"Subfolder not found: {subfolder_path}")
            continue
            
        logger.info(f"Processing JSON files in subfolder: {subfolder}")
        
        # Find all JSON files in the subfolder
        json_files = glob.glob(os.path.join(subfolder_path, '*.json'))
        
        if not json_files:
            logger.info(f"No JSON files found in '{subfolder_path}'.")
        else:
            logger.info(f"Found {len(json_files)} JSON files to process in {subfolder}.")
            all_files_count += len(json_files)
            
            # Process each file
            for file_path in json_files:
                try:
                    # Skip job_info.json files
                    if os.path.basename(file_path) == 'job_info.json':
                        skipped_count += 1
                        continue
                    
                    # Check if file is already in the expected format
                    already_processed = False
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list) and all(isinstance(item, dict) and 'page' in item and 'md' in item for item in data):
                                logger.info(f"File {os.path.basename(file_path)} is already in the expected format. Skipping.")
                                already_processed = True
                                skipped_count += 1
                    except:
                        # If we can't open or parse the file, we'll try processing it
                        pass
                    
                    if not already_processed:
                        if llamaparse_processing.process_file(file_path):
                            success_count += 1
                        else:
                            error_count += 1
                except Exception as e:
                    logger.error(f"CRITICAL ERROR processing file {os.path.basename(file_path)}: {e}")
                    error_count += 1
    
    # Summary
    logger.info(f"Post-processing summary - Total files: {all_files_count}, Successful: {success_count}, Skipped: {skipped_count}, Failed: {error_count}")
    
    # Consider success even if all files were skipped (already processed)
    return {
        "success": (success_count > 0) or (skipped_count > 0),
        "all_files_count": all_files_count,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "error_count": error_count
    }

def process_json_with_llm(json_output_dir):
    """
    Process parsed JSON files using LLM to extract structured data from markdown content.
    
    Args:
        json_output_dir: Base directory containing the parsed JSON results
        
    Returns:
        dict: Result summary with success and failure counts
    """
    logger.info("==== Processing JSON files with LLM for structured data extraction ====")
    
    # Run the async main function from llamaparse_processing
    asyncio.run(llamaparse_processing.main_async())
    
    # Run the summary generation function from post_processing instead of llamaparse_processing
    logger.info("==== Generating summaries for processed content ====")
    asyncio.run(post_processing.generate_summaries_main_async())
    
    # Return success indicator (actual counts are logged in llamaparse_processing)
    return {
        "success": True  # Assuming success if no exceptions
    }

def final_post_process_json(json_output_dir):
    """
    Perform final post-processing on JSON files for ISM Manual and Office Manual to flatten
    and structure the data properly.
    
    Args:
        json_output_dir: Base directory containing the processed JSON results
        
    Returns:
        dict: Result summary with success and failure counts
    """
    logger.info("==== Performing final JSON post-processing for flattening and structuring ====")
    
    # Run the main function from post_processing
    post_processing.main()
    
    # Return success indicator
    return {
        "success": True  # Assuming success if no exceptions
    }

def update_json_with_metadata(json_output_dir, csv_file_path):
    """
    Updates all JSON files with metadata from the CSV file.
    
    Args:
        json_output_dir: Base directory containing the parsed JSON results
        csv_file_path: Path to the CSV file with document metadata
        
    Returns:
        dict: Result summary with success and failure counts
    """
    logger.info("==== Starting JSON Metadata Update ====")
    
    # CSV Column Names
    csv_doc_no_col = "Doc. No"
    csv_doc_title_col = "Document title"
    csv_doc_type_col = "Type"
    csv_doc_id_col = "DocID"

    # JSON Field Names
    json_doc_no_field = "docNo"
    json_doc_title_field = "documentTitle"
    json_doc_type_field = "type"
    json_doc_id_field = "docId"
    json_doc_header_field = "documentHeader"
    json_embed_text_field = "embText"
    
    # Load document metadata from CSV
    doc_map = {}
    try:
        logger.info(f"Loading document metadata from CSV: {csv_file_path}")
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            header = reader.fieldnames
            if not header:
                logger.error("CSV file is empty or has no header row.")
                return {"success": False, "error": "CSV file is empty or has no header"}
            
            required_cols = [csv_doc_no_col, csv_doc_title_col, csv_doc_type_col, csv_doc_id_col]
            missing_cols = [col for col in required_cols if col not in header]
            if missing_cols:
                logger.error(f"CSV file is missing required columns: {', '.join(missing_cols)}. Required: {required_cols}")
                return {"success": False, "error": f"CSV file is missing required columns: {', '.join(missing_cols)}"}

            # Process rows
            row_count = 0
            skipped_rows = 0
            for row in reader:
                row_count += 1
                doc_id = row.get(csv_doc_id_col, '').strip()
                doc_no = row.get(csv_doc_no_col, '').strip()
                doc_title = row.get(csv_doc_title_col, '').strip()
                doc_type = row.get(csv_doc_type_col, '').strip()

                # Store metadata by DocID
                if doc_id:
                    doc_map[doc_id] = {
                        json_doc_no_field: doc_no if doc_no else 'N/A',
                        json_doc_title_field: doc_title if doc_title else 'N/A',
                        json_doc_type_field: doc_type if doc_type else 'N/A',
                        json_doc_id_field: doc_id
                    }
                else:
                    skipped_rows += 1
                    
            logger.info(f"Loaded metadata for {len(doc_map)} documents from {row_count} rows (skipped {skipped_rows} rows).")
            
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
        return {"success": False, "error": f"CSV file not found: {csv_file_path}"}
    except Exception as e:
        logger.exception(f"Error loading document metadata: {e}")
        return {"success": False, "error": f"Error loading document metadata: {e}"}
    
    if not doc_map:
        logger.error("No document metadata loaded from CSV file.")
        return {"success": False, "error": "No document metadata loaded"}
    
    # Process JSON files recursively
    processed_files = 0
    skipped_files = 0
    
    # Helper function to update a single JSON item with metadata and required fields
    def update_json_item(item, csv_data):
        """
        Updates a single JSON dictionary with metadata fields and required fields.
        Returns True if changes were made, False otherwise.
        """
        try:
            changes_made = False
            
            # Always remove keywords field if present
            if 'keywords' in item:
                del item['keywords']
                changes_made = True
                
            # Add metadata fields from CSV
            item[json_doc_no_field] = csv_data[json_doc_no_field]
            item[json_doc_title_field] = csv_data[json_doc_title_field]
            item[json_doc_type_field] = csv_data[json_doc_type_field]
            item[json_doc_id_field] = csv_data[json_doc_id_field]
            
            # Get values for additional fields, using empty strings for missing values
            doc_name = item.get('documentName', '')
            doc_title = item.get(json_doc_title_field, '')
            section = item.get('section', '')
            chapter = item.get('chapter', '')
            doc_type = item.get(json_doc_type_field, '')
            original_text = item.get('originalText', '')
            
            # Convert page to pageNumber if needed
            page = item.get('page', '')
            if 'page' in item:
                item['pageNumber'] = item['page']
                if 'pageNumber' not in item:  # Only delete if we successfully copied
                    del item['page']
            elif 'pageNumber' not in item:
                item['pageNumber'] = ''
            
            # Create documentHeader field combining available fields
            header_parts = []
            if doc_name: header_parts.append(f"Document name: {doc_name}")
            if doc_title: header_parts.append(f"Document title: {doc_title}")
            if section: header_parts.append(f"Section name: {section}")
            if chapter: header_parts.append(f"Chapter name: {chapter}")
            if doc_type: header_parts.append(f"Type: {doc_type}")
            if item['pageNumber']: header_parts.append(f"Page number: {item['pageNumber']}")
            
            # If header_parts is empty, add at least document title and type to avoid empty headers
            if not header_parts:
                if doc_title: header_parts.append(f"Document title: {doc_title}")
                if doc_type: header_parts.append(f"Type: {doc_type}")
                logger.warning(f"  Created minimal header with only title/type as no other header fields were available")
            
            document_header = " | ".join(header_parts)
            item[json_doc_header_field] = document_header
            
            # Create embText field combining available fields plus original text
            embed_parts = header_parts.copy()
            if original_text: embed_parts.append(f"Original text: {original_text}")
            
            embed_text = " | ".join(embed_parts)
            item[json_embed_text_field] = embed_text
            
            # Add default required fields if missing
            required_fields = {
                "chunkNo": 1,  # Default to 1 if missing
                "revDate": "",
                "reviewDate": "",
                "chapter": "",
                "documentName": "",
                "section": "",
                "DOC": "SMPL",
                "identifier": "SMPL",
                "summary": "",
                "embType": "",
                "sourceId": "",
                "subSection": "",
                "type": doc_type,
                "documentTitle": doc_title,
                "shortSummary": "",
                "docNo": item.get(json_doc_no_field, ""),
                "docId": item.get(json_doc_id_field, ""),
            }
            
            # For non-ISM and non-Office Manual documents, copy originalText to shortSummary
            is_ism_or_office = doc_type in ["ISM Manual", "Office Manual"]
            if not is_ism_or_office and original_text and not item.get('shortSummary'):
                item['shortSummary'] = original_text
            
            # Add any missing fields
            for field, default_value in required_fields.items():
                if field not in item:
                    item[field] = default_value
                    
            return True
        except Exception as e:
            logger.error(f"  Error updating JSON item: {e}", exc_info=True)
            return False
    
    # Helper function to add only documentHeader and embText when CSV data is not available
    def add_document_header_and_embed_text(item):
        """
        Adds only documentHeader and embText fields to a JSON item.
        Returns True if changes were made, False otherwise.
        """
        try:
            changes_made = False
            
            # Always remove keywords field if present
            if 'keywords' in item:
                del item['keywords']
                changes_made = True
                
            # Get values for fields needed to create header and embText
            doc_name = item.get('documentName', '')
            doc_title = item.get('documentTitle', '')
            section = item.get('section', '')
            chapter = item.get('chapter', '')
            doc_type = item.get('type', '')
            original_text = item.get('originalText', '')
            
            # Convert page to pageNumber if needed
            if 'page' in item:
                item['pageNumber'] = item['page']
                if 'pageNumber' not in item:  # Only delete if we successfully copied
                    del item['page']
            elif 'pageNumber' not in item:
                item['pageNumber'] = ''
            
            # Create documentHeader field combining available fields
            header_parts = []
            if doc_name: header_parts.append(f"Document name: {doc_name}")
            if doc_title: header_parts.append(f"Document title: {doc_title}")
            if section: header_parts.append(f"Section name: {section}")
            if chapter: header_parts.append(f"Chapter name: {chapter}")
            if doc_type: header_parts.append(f"Type: {doc_type}")
            if item['pageNumber']: header_parts.append(f"Page number: {item['pageNumber']}")
            
            # If header_parts is empty, try the documentHeader if it exists already
            if not header_parts and json_doc_header_field in item:
                document_header = item[json_doc_header_field]
            else:
                document_header = " | ".join(header_parts)
                item[json_doc_header_field] = document_header
            
            # Create embText field combining available fields plus original text
            embed_parts = []
            if document_header: 
                embed_parts.append(document_header)
            if original_text: 
                embed_parts.append(f"Original text: {original_text}")
            
            embed_text = " | ".join(embed_parts)
            item[json_embed_text_field] = embed_text
            
            # Add default required fields if missing
            required_fields = {
                "chunkNo": 1,  # Default to 1 if missing
                "revDate": "",
                "reviewDate": "",
                "chapter": "",
                "documentName": "",
                "section": "",
                "DOC": "SMPL",
                "identifier": "SMPL",
                "summary": "",
                "embType": "",
                "sourceId": "",
                "subSection": "",
                "type": doc_type,
                "documentTitle": doc_title,
                "shortSummary": "",
                "docNo": "",
                "docId": "",
            }
            
            # For non-ISM and non-Office Manual documents, copy originalText to shortSummary
            is_ism_or_office = doc_type in ["ISM Manual", "Office Manual"]
            if not is_ism_or_office and original_text and not item.get('shortSummary'):
                item['shortSummary'] = original_text
            
            # Add any missing fields
            for field, default_value in required_fields.items():
                if field not in item:
                    item[field] = default_value
            
            return True
        except Exception as e:
            logger.error(f"  Error adding documentHeader and embText: {e}", exc_info=True)
            return False
    
    # Helper function to process a single directory
    def process_directory(directory):
        nonlocal processed_files, skipped_files
        
        try:
            directory_path = Path(directory)
            logger.info(f"Processing directory: {directory_path}")
            
            # Process all items in the directory
            for item in directory_path.iterdir():
                if item.is_dir():
                    # Recursively process subdirectories
                    process_directory(item)
                elif item.is_file() and item.suffix.lower() == '.json':
                    # Process JSON file
                    filename = item.name
                    
                    try:
                        # Extract DocID from filename
                        match = re.search(r'(\d+)_', filename)
                        if not match:
                            logger.warning(f"Skipping file {filename}: Could not extract DocID from filename.")
                            skipped_files += 1
                            continue
                            
                        doc_id = match.group(1)
                        
                        # Read the JSON file
                        with open(item, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                logger.warning(f"Skipping file {filename}: File is empty.")
                                skipped_files += 1
                                continue
                                
                            data = json.loads(content)
                        
                        # Look up data in the CSV map
                        csv_data = doc_map.get(doc_id)
                        process_full_metadata = True
                        
                        if not csv_data:
                            logger.warning(f"DocID '{doc_id}' not found in the provided CSV map. Will only process documentHeader and embText fields.")
                            process_full_metadata = False
                            # Create a minimal csv_data structure with empty values
                            csv_data = {
                                json_doc_no_field: 'N/A',
                                json_doc_title_field: '',
                                json_doc_type_field: '',
                                json_doc_id_field: doc_id
                            }
                        
                        # Track if any changes were made to the file
                        changes_made = False
                        
                        # Process the data based on its structure
                        if isinstance(data, dict):
                            # Process a single dictionary object
                            if process_full_metadata:
                                if update_json_item(data, csv_data):
                                    changes_made = True
                            else:
                                if add_document_header_and_embed_text(data):
                                    changes_made = True
                                    
                            # Verify the fields were added and retained
                            if json_doc_header_field not in data or json_embed_text_field not in data:
                                logger.warning(f"Some fields missing after update: documentHeader={json_doc_header_field in data}, embText={json_embed_text_field in data}")
                        elif isinstance(data, list):
                            # Process a list of items
                            items_updated = 0
                            for i, item_dict in enumerate(data):
                                if isinstance(item_dict, dict):
                                    if process_full_metadata:
                                        if update_json_item(item_dict, csv_data):
                                            changes_made = True
                                            items_updated += 1
                                    else:
                                        if add_document_header_and_embed_text(item_dict):
                                            changes_made = True
                                            items_updated += 1
                                            
                                    # Verify for a sample of items (first, last, and middle)
                                    if i == 0 or i == len(data)-1 or i == len(data)//2:
                                        if json_doc_header_field not in item_dict or json_embed_text_field not in item_dict:
                                            logger.warning(f"Item {i}: Some fields missing after update: documentHeader={json_doc_header_field in item_dict}, embText={json_embed_text_field in item_dict}")
                            logger.info(f"Updated {items_updated} of {len(data)} items")
                        else:
                            logger.warning(f"Skipping file {filename}: JSON content is neither a dictionary nor a list.")
                            skipped_files += 1
                            continue
                            
                        # Write the updated JSON data back to the file if changes were made
                        if changes_made:
                            with open(item, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=4, ensure_ascii=False)
                                
                            logger.info(f"Successfully updated metadata for: {filename}")
                            processed_files += 1
                        else:
                            logger.info(f"No changes needed for: {filename}")
                            processed_files += 1
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from file: {filename}. Error: {e}")
                        skipped_files += 1
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while processing {filename}: {e}")
                        skipped_files += 1
                        
        except Exception as e:
            logger.exception(f"Error processing directory {directory}: {e}")
    
    # Start processing from the root directory
    try:
        start_time = time.time()
        process_directory(json_output_dir)
        end_time = time.time()
        
        logger.info("--- Metadata Update Summary ---")
        logger.info(f"Total files processed: {processed_files + skipped_files}")
        logger.info(f"Files successfully updated: {processed_files}")
        logger.info(f"Files skipped: {skipped_files}")
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
        
        return {
            "success": processed_files > 0,
            "processed_files": processed_files,
            "skipped_files": skipped_files
        }
        
    except Exception as e:
        logger.exception(f"Error during metadata update: {e}")
        return {"success": False, "error": f"Error during metadata update: {e}"}

def add_chunk_numbers_to_json(json_output_dir):
    """
    Add chunk numbers to all JSON files in the specified directories.
    Uses the add_chunk_numbers function from post_processing.
    
    Args:
        json_output_dir (str): Base directory where JSON files are stored
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    logger.info("==== Adding Chunk Numbers to JSON Files ====")
    start_time = time.time()
    
    # Run the async function for adding chunk numbers from post_processing instead of llamaparse_processing
    try:
        asyncio.run(post_processing.process_chunk_numbers_main_async())
        success = True
    except Exception as e:
        logger.exception(f"Error adding chunk numbers to JSON files: {e}")
        success = False
    
    end_time = time.time()
    logger.info(f"Time taken to add chunk numbers: {end_time - start_time:.2f} seconds")
    
    return success

def sync_to_typesense():
    """
    Sync MongoDB documents to Typesense regardless of previous upload status.
    This function ensures Typesense indexing happens even if there were issues with MongoDB upload.
    
    Returns:
        dict: Result summary with success status and count of documents imported
    """
    logger.info("Starting Typesense sync to index MongoDB documents...")
    try:
        typesense_result = mongodb_to_typesense.sync_mongodb_to_typesense()
        if typesense_result.get('success', False):
            logger.info(f"Typesense sync completed successfully: {typesense_result.get('total_docs_imported', 0)} documents imported.")
        else:
            logger.error(f"Typesense sync failed: {typesense_result.get('error', 'Unknown error')}")
        return typesense_result
    except Exception as typesense_error:
        logger.exception(f"Error during Typesense sync: {typesense_error}")
        return {"success": False, "error": str(typesense_error)}

def cleanup_files(output_dir, json_output_dir, log_file):
    """
    Delete all files and directories created during processing except for the log file.
    
    Args:
        output_dir: Directory containing downloaded files
        json_output_dir: Directory containing parsed JSON results
        log_file: Log file to preserve
    """
    logger.info("==== Starting cleanup of created files and directories ====")
    
    # Delete output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        try:
            logger.info(f"Removing output directory: {output_dir}")
            shutil.rmtree(output_dir)
            logger.info(f"Successfully removed {output_dir}")
        except Exception as e:
            logger.error(f"Error removing output directory {output_dir}: {e}")
    
    # Delete JSON output directory
    if os.path.exists(json_output_dir) and os.path.isdir(json_output_dir):
        try:
            logger.info(f"Removing JSON output directory: {json_output_dir}")
            shutil.rmtree(json_output_dir)
            logger.info(f"Successfully removed {json_output_dir}")
        except Exception as e:
            logger.error(f"Error removing JSON output directory {json_output_dir}: {e}")
    
    # Delete the downloaded zip file
    zip_file_path = f"{output_dir}.zip"
    if os.path.exists(zip_file_path):
        try:
            logger.info(f"Removing downloaded zip file: {zip_file_path}")
            os.remove(zip_file_path)
            logger.info(f"Successfully removed {zip_file_path}")
        except Exception as e:
            logger.error(f"Error removing zip file {zip_file_path}: {e}")
    
    # Delete all other files in the current directory except the log file
    current_dir = os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else '.'
    logger.info(f"Cleaning up files in current directory: {current_dir}")
    
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        
        # Skip the log file
        if item_path == os.path.abspath(log_file):
            continue
        
        # Skip directories we've already handled
        if item_path == os.path.abspath(output_dir) or item_path == os.path.abspath(json_output_dir):
            continue
            
        try:
            if os.path.isdir(item_path):
                # For CSV directory or other directories created during processing
                if "docmap" in item.lower() or "csv" in item.lower():
                    logger.info(f"Removing directory: {item_path}")
                    shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                # For temporary files created during processing
                if item.endswith(".csv") or item.endswith(".json") or item.endswith(".zip") or "docmap" in item.lower():
                    logger.info(f"Removing file: {item_path}")
                    os.remove(item_path)
        except Exception as e:
            logger.error(f"Error removing {item_path}: {e}")
    
    logger.info("==== Cleanup complete ====")

def main():
    """Main function that orchestrates the entire automation process."""
    start_time = datetime.now()
    logger.info("==== Starting Document Automation ====")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure JSON output directory exists and is absolute
    if not os.path.isabs(args.json_output_dir):
        args.json_output_dir = os.path.abspath(args.json_output_dir)
    logger.info(f"JSON output directory set to: {args.json_output_dir}")
    os.makedirs(args.json_output_dir, exist_ok=True)
    
    # Determine which document types to process and create directories
    document_types_to_process = []
    
    # Set default behavior - process all document types when no specific flags are set
    no_specific_flags = not any([args.only_scrape, args.only_convert, args.only_parse, 
                                args.parse_ism_only, args.parse_doc_type, args.only_add_chunks])
    
    # Determine if we're processing ISM Manual or Office Manual
    if args.parse_ism_only or (args.parse_doc_type == "ISM Manual"):
        document_types_to_process.append("ISM Manual")
    elif args.parse_doc_type == "Office Manual":
        document_types_to_process.append("Office Manual")
    elif args.parse_doc_type == "all" or no_specific_flags:
        # All includes both manual types and general types
        document_types_to_process.extend(["ISM Manual", "Office Manual", "Policies", "Sample Document Type"])
        document_types_to_process.extend(document_parser_general.DOCUMENT_TYPES)
    elif args.parse_doc_type in document_parser_general.DOCUMENT_TYPES:
        document_types_to_process.append(args.parse_doc_type)
    elif args.only_parse and not args.parse_doc_type:
        # Default behavior with just --only-parse
        document_types_to_process.extend(["ISM Manual", "Office Manual", "Policies", "Sample Document Type"])
        document_types_to_process.extend(document_parser_general.DOCUMENT_TYPES)
    
    # Check which document types actually have documents before creating directories
    verified_document_types = []
    for doc_type in document_types_to_process:
        # For output directory, always create regardless of input
        output_doc_dir = get_document_type_dir(args.json_output_dir, doc_type, create_if_missing=True)
        
        # Only process types with existing documents in the input directory
        input_doc_dir = get_document_type_dir(args.output_dir, doc_type, create_if_missing=False)
        if input_doc_dir:
            # Check if directory contains any files
            has_files = False
            for root, _, files in os.walk(input_doc_dir):
                if files:
                    has_files = True
                    break
            
            if has_files:
                verified_document_types.append(doc_type)
                logger.info(f"Found documents for {doc_type} - will process this type")
            else:
                logger.info(f"No documents found for {doc_type} - skipping this type")
        else:
            logger.info(f"No input directory found for {doc_type} - skipping this type")
    
    # Update the list of types to process to only include those with documents
    document_types_to_process = verified_document_types
    
    scrape_success = True
    convert_success = True
    parse_success = True
    post_process_success = True
    chunk_numbers_success = True
    
    # If only adding chunk numbers, skip other steps
    if args.only_add_chunks:
        logger.info("==== Only Adding Chunk Numbers to JSON Files ====")
        chunk_numbers_success = add_chunk_numbers_to_json(args.json_output_dir)
    else:
        # Step 1: Scrape and download documents (unless --only-convert, --only-parse, etc. is specified)
        if not any([args.only_convert, args.only_parse, args.parse_ism_only, args.parse_doc_type]):
            logger.info("==== Starting Document Scraping ====")
            try:
                scrape_success = document_scrap.scrape_documents(
                    username=args.username,
                    password=args.password,
                    domain=args.domain,
                    output_dir=args.output_dir
                )
                
                if not scrape_success:
                    logger.error("Document scraping failed. See log for details.")
                    if not args.only_scrape:
                        logger.info("Attempting to proceed with subsequent steps despite scraping error...")
                else:
                    logger.info("Document scraping completed successfully!")
                    
            except Exception as e:
                logger.exception(f"Unhandled exception during document scraping: {e}")
                scrape_success = False
                if not args.only_scrape:
                    logger.info("Attempting to proceed with subsequent steps despite scraping error...")
        
        # Step 2: Convert files to PDF (unless --only-scrape, --only-parse, etc. is specified)
        if not any([args.only_scrape, args.only_parse, args.parse_ism_only, args.parse_doc_type]):
            logger.info("==== Starting PDF Conversion ====")
            try:
                conversion_result = document_to_pdf.convert_to_pdf(
                    client_id=args.adobe_id,
                    client_secret=args.adobe_secret,
                    input_directory=args.output_dir,
                    max_workers=args.max_workers
                )
                
                convert_success = conversion_result.get('success', False)
                if not convert_success:
                    logger.error(f"PDF conversion failed: {conversion_result.get('error', 'Unknown error')}")
                else:
                    total_files = conversion_result.get('total_files', 0)
                    success_count = conversion_result.get('success_count', 0)
                    error_count = conversion_result.get('error_count', 0)
                    
                    logger.info(f"PDF conversion completed: {success_count} of {total_files} files converted successfully.")
                    if error_count > 0:
                        logger.warning(f"{error_count} file(s) failed to convert.")
                        for failed_file in conversion_result.get('failed_files', []):
                            logger.warning(f"Failed to convert: {failed_file.get('input')} - {failed_file.get('message')}")
                            
            except Exception as e:
                logger.exception(f"Unhandled exception during PDF conversion: {e}")
                convert_success = False
        
        # Step 3: Parse documents to JSON (if requested)
        if args.only_parse or args.parse_ism_only or args.parse_doc_type is not None or (not any([args.only_scrape, args.only_convert])):
            logger.info("==== Starting Document Parsing ====")
            
            # Determine which document types to parse
            parse_manual_types = []
            parse_general_types = []
            
            if args.parse_ism_only:
                # Only parse ISM documents
                parse_manual_types = ["ISM Manual"]
            elif args.parse_doc_type == "all" or no_specific_flags:
                # Parse all document types (both manual types and general types)
                parse_manual_types = ["ISM Manual", "Office Manual", "Policies", "Sample Document Type"]
                parse_general_types = document_parser_general.DOCUMENT_TYPES
            elif args.parse_doc_type in ["ISM Manual", "Office Manual", "Policies", "Sample Document Type"]:
                # Parse specific manual document type
                parse_manual_types = [args.parse_doc_type]
            elif args.parse_doc_type is not None:
                # Parse specific general document type
                parse_general_types = [args.parse_doc_type]
            else:
                # Default behavior with --only-parse and no specific type
                parse_manual_types = ["ISM Manual", "Office Manual", "Policies", "Sample Document Type"]
                parse_general_types = document_parser_general.DOCUMENT_TYPES
            
            # Parse manual documents (ISM Manual and Office Manual)
            manual_parsing_success = True
            if parse_manual_types:
                for doc_type in parse_manual_types:
                    try:
                        logger.info(f"Parsing {doc_type} documents...")
                        parse_result = parse_manual_documents(
                            input_dir=args.output_dir,
                            output_dir=args.json_output_dir,
                            document_type=doc_type
                        )
                        
                        doc_success = parse_result.get('success', False)
                        if not doc_success:
                            logger.error(f"{doc_type} parsing failed: {parse_result.get('error', 'Unknown error')}")
                            manual_parsing_success = False
                        else:
                            logger.info(f"{doc_type} parsing completed: {parse_result.get('success_count', 0)} files parsed successfully.")
                        
                    except Exception as e:
                        logger.exception(f"Unhandled exception during {doc_type} parsing: {e}")
                        manual_parsing_success = False
            
            # Parse other document types
            general_parsing_success = True
            if parse_general_types:
                try:
                    for doc_type in parse_general_types:
                        logger.info(f"Parsing documents of type: {doc_type}")
                        parse_doc_result = parse_documents_by_type(
                            input_dir=args.output_dir,
                            output_dir=args.json_output_dir,
                            doc_type=doc_type,
                            max_workers=args.max_workers,
                            openai_api_key=args.openai_api_key
                        )
                        
                        # Consider it a success even if there were no files to process 
                        # (we mark it as success in parse_documents_by_type for missing directories)
                        doc_type_success = parse_doc_result.get('success', False)
                        if not doc_type_success:
                            logger.error(f"Document type '{doc_type}' parsing failed: {parse_doc_result.get('error', 'Unknown error')}")
                            general_parsing_success = False
                        else:
                            processed = parse_doc_result.get('processed', 0)
                            skipped = parse_doc_result.get('skipped', 0)
                            if processed > 0:
                                logger.info(f"Document type '{doc_type}' parsing completed: {processed} files parsed successfully.")
                            if skipped > 0:
                                logger.info(f"Document type '{doc_type}' parsing: {skipped} files skipped (already parsed).")
                
                except Exception as e:
                    logger.exception(f"Unhandled exception during document type parsing: {e}")
                    general_parsing_success = False
            
            # Overall parsing success
            parse_success = manual_parsing_success and general_parsing_success
            
            # Step 4: Post-process parsed JSON files for ISM and Office Manual
            if parse_manual_types and manual_parsing_success:
                try:
                    logger.info("Starting post-processing of parsed JSON files...")
                    post_process_result = process_manual_json_files(args.json_output_dir)
                    post_process_success = post_process_result.get('success', False)
                    
                    if not post_process_success:
                        logger.warning("Initial JSON post-processing reported no successful files. This may be OK if files were already processed.")
                    
                    # Always proceed to LLM processing regardless of initial post-processing results
                    # Step 5: Process JSON files with LLM to extract structured data
                    logger.info("Starting LLM-based structured data extraction...")
                    llm_process_success = False
                    try:
                        llm_process_result = process_json_with_llm(args.json_output_dir)
                        llm_process_success = llm_process_result.get('success', False)
                        
                        if not llm_process_success:
                            logger.error("LLM-based data extraction failed")
                        else:
                            logger.info("LLM-based data extraction completed successfully.")
                            
                            # Step 6: Final post-processing to flatten and structure JSON
                            logger.info("Starting final post-processing for flattening and structuring...")
                            try:
                                final_post_process_result = final_post_process_json(args.json_output_dir)
                                final_post_process_success = final_post_process_result.get('success', False)
                                
                                if not final_post_process_success:
                                    logger.error("Final JSON post-processing failed")
                                else:
                                    logger.info("Final JSON post-processing completed successfully.")
                                    
                                    # Step 7: Update JSON files with metadata from CSV
                                    logger.info("Starting JSON metadata update...")
                                    try:
                                        csv_file_path = os.path.join(os.path.dirname(args.json_output_dir), "docmap_final_processed.csv")
                                        metadata_update_result = update_json_with_metadata(args.json_output_dir, csv_file_path)
                                        metadata_update_success = metadata_update_result.get('success', False)
                                        
                                        if not metadata_update_success:
                                            logger.error(f"JSON metadata update failed: {metadata_update_result.get('error', 'Unknown error')}")
                                        else:
                                            logger.info(f"JSON metadata update completed successfully: {metadata_update_result.get('processed_files', 0)} files updated.")
                                            
                                            # Call document_s3_uploader after metadata update 
                                            # Modified to run by default or when --parse-doc-type all is specified
                                            if args.parse_doc_type == "all" or no_specific_flags or (args.only_parse and not args.parse_doc_type):
                                                logger.info("Starting S3 upload for document files and updating JSON with links...")
                                                try:
                                                    s3_upload_result = document_s3_uploader.process_documents()
                                                    if s3_upload_result.get('success', False):
                                                        logger.info(f"S3 upload completed successfully: {s3_upload_result.get('updated_json_files', 0)} JSON files updated with document links.")
                                                        
                                                        # Insert documents into MongoDB and add MongoDB IDs to JSON files
                                                        logger.info("Starting MongoDB upload and adding MongoDB IDs to JSON files...")
                                                        try:
                                                            mongo_result = upload_to_mongodb.upload_documents(
                                                                json_folder_path=args.json_output_dir
                                                            )
                                                            if mongo_result.get('success', False):
                                                                logger.info(f"MongoDB upload completed successfully: {mongo_result.get('documents_inserted', 0)} documents inserted, {mongo_result.get('files_updated_with_ids', 0)} files updated with MongoDB IDs.")
                                                                
                                                                # Call Typesense sync regardless of MongoDB upload status
                                                                sync_to_typesense()
                                                        except Exception as mongo_error:
                                                                logger.exception(f"Error during MongoDB upload: {mongo_error}")
                                                                # Still try to sync with Typesense even if MongoDB had an exception
                                                                sync_to_typesense()
                                                    else:
                                                        logger.error("S3 upload and JSON update failed.")
                                                except Exception as s3_error:
                                                    logger.exception(f"Error during S3 upload and JSON update: {s3_error}")
                                                    
                                    except Exception as e:
                                        logger.exception(f"Unhandled exception during JSON metadata update: {e}")
                                        
                            except Exception as e:
                                logger.exception(f"Unhandled exception during final JSON post-processing: {e}")
                                
                    except Exception as e:
                        logger.exception(f"Unhandled exception during LLM-based extraction: {e}")
                    
                except Exception as e:
                    logger.exception(f"Unhandled exception during JSON post-processing: {e}")
                    post_process_success = False
            
            # Step 6: Add chunk numbers to processed JSON files
            if parse_success and post_process_success:
                logger.info("==== Adding Chunk Numbers to JSON Files ====")
                chunk_numbers_success = add_chunk_numbers_to_json(args.json_output_dir)
                if not chunk_numbers_success:
                    logger.error("Failed to add chunk numbers to JSON files. See log for details.")
        
        # Final steps for general document types
        general_doc_types = [dt for dt in document_types_to_process if dt in document_parser_general.DOCUMENT_TYPES]
        if general_doc_types:
            # Step 6: Add chunk numbers to general document types
            logger.info("==== Adding Chunk Numbers to General Document Type JSON Files ====")
            chunk_numbers_success = add_chunk_numbers_to_json(args.json_output_dir)
            if not chunk_numbers_success:
                logger.error("Failed to add chunk numbers to JSON files. See log for details.")
            
            # Also run S3 upload for general document types
            # Modified to run by default or when --parse-doc-type all is specified
            if args.parse_doc_type == "all" or no_specific_flags or (args.only_parse and not args.parse_doc_type):
                logger.info("==== Starting S3 Upload for General Document Types ====")
                try:
                    s3_upload_result = document_s3_uploader.process_documents()
                    if s3_upload_result.get('success', False):
                        logger.info(f"S3 upload completed successfully: {s3_upload_result.get('updated_json_files', 0)} JSON files updated with document links.")
                        
                        # Insert documents into MongoDB and add MongoDB IDs to JSON files
                        logger.info("Starting MongoDB upload and adding MongoDB IDs to JSON files...")
                        try:
                            mongo_result = upload_to_mongodb.upload_documents(
                                json_folder_path=args.json_output_dir
                            )
                            if mongo_result.get('success', False):
                                logger.info(f"MongoDB upload completed successfully: {mongo_result.get('documents_inserted', 0)} documents inserted, {mongo_result.get('files_updated_with_ids', 0)} files updated with MongoDB IDs.")
                                
                                # Call Typesense sync regardless of MongoDB upload status
                                sync_to_typesense()
                            else:
                                logger.error(f"MongoDB upload failed: {mongo_result.get('error', 'Unknown error')}")
                        except Exception as mongo_error:
                            logger.exception(f"Error during MongoDB upload: {mongo_error}")
                            # Still try to sync with Typesense even if MongoDB had an exception
                            sync_to_typesense()
                    else:
                        logger.error("S3 upload and JSON update failed.")
                except Exception as s3_error:
                    logger.exception(f"Error during S3 upload and JSON update: {s3_error}")
            
    # Generate final output summary CSV
    # ... existing code for CSV generation ...

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info("==== Document Automation Complete ====")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {elapsed_time}")
    
    # Always clean up all files and directories, regardless of success
    logger.info("Cleaning up all created files and directories, preserving only log files.")
    #cleanup_files(args.output_dir, args.json_output_dir, log_filename)
    
    return 0

if __name__ == "__main__":
    logger.info(f"Starting document automation with log file: {log_filename}")
    try:
        exit_code = main()
        logger.info(f"Process completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Fatal error occurred: {e}", exc_info=True)
        sys.exit(1) 