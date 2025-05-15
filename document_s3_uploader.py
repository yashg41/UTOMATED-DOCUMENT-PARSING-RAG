#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document S3 Uploader

This script uploads PDF and non-PDF documents to S3 and updates the corresponding JSON files with links.
It handles different document types (ISM Manual, Office Manual, Fleet Alert, etc.) and organizes files
by document type in separate folders on S3.

The script:
1. Identifies document types from JSON files
2. Finds the corresponding PDF and original files in the documents directory
3. Uploads these files to appropriate S3 folders based on document type
4. Updates the JSON files with the S3 URLs for both PDF and original files
"""

import boto3
import botocore
import os
import json
import urllib.parse
import logging
import sys
import time
import re
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# First check if env vars are set
env_aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
env_aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
env_aws_region = os.getenv("AWS_REGION", "ap-south-1")
env_bucket_name = os.getenv("S3_BUCKET_NAME", "sm2.0-etl-prod-ap-south-1-274743989443")

# Debug information about environment variables
print("---- AWS CREDENTIALS DEBUG INFO ----")
print(f"AWS_ACCESS_KEY_ID environment variable set: {'Yes' if env_aws_key_id else 'No'}")
print(f"AWS_SECRET_ACCESS_KEY environment variable set: {'Yes' if env_aws_secret_key else 'No'}")
print(f".env file exists in current directory: {'Yes' if os.path.exists('.env') else 'No'}")
if env_aws_key_id:
    masked_key_id = f"{env_aws_key_id[:5]}...{env_aws_key_id[-4:]}" if len(env_aws_key_id) > 10 else "***masked***"
    print(f"Using AWS_ACCESS_KEY_ID from environment: {masked_key_id}")
else:
    print("WARNING: AWS_ACCESS_KEY_ID not found in environment variables!")

# Set variables with or without defaults
AWS_ACCESS_KEY_ID = env_aws_key_id
AWS_SECRET_ACCESS_KEY = env_aws_secret_key
AWS_REGION = env_aws_region
BUCKET_NAME = env_bucket_name
print("---------------------------")

# --- Directory and File Configuration ---
JSON_BASE_DIR = "parsed_json_results"
SOURCE_FILES_DIR = "documents"
JSON_SUFFIX = '_result.json'
ALLOWED_ORIGINAL_EXTENSIONS = ['.docx', '.xlsx', '.doc', '.xls', '.ppt', '.pptx']

# --- S3 Base Configuration ---
S3_BASE_FOLDER = "syia_documents"

# --- Document Type Configuration ---
DOCUMENT_TYPES = [
    "Fleet Alert",
    "Standard eForm",
    "Forms and Checklists",
    "RA Database", 
    "Office Forms",
    "ISM Manual", 
    "Office Manual",
    "Policies",
    "Sample Document Type"
]

# Map of document types to S3 folder names
DOCUMENT_TYPE_TO_S3_FOLDER = {
    "Fleet Alert": "Fleet_Alert",
    "Standard eForm": "Standard_eForm",
    "Forms and Checklists": "Forms_and_Checklists",
    "RA Database": "RA_Database",
    "Office Forms": "Office_Forms",
    "ISM Manual": "ISM_Manual",
    "Office Manual": "Office_Manual",
    "Policies": "Policies",
    "Sample Document Type": "Sample_Document_Type"
}

# --- Logging Configuration ---
log_filename = f"document_s3_upload_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_filename}")

# --- Initialize S3 Client ---
try:
    s3_client = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    logger.info(f"Successfully initialized S3 client for region {AWS_REGION}.")
    print(f"Successfully initialized S3 client for region {AWS_REGION}.")
except botocore.exceptions.NoCredentialsError:
    logger.error("AWS credentials not found.")
    print("Error: AWS credentials not found.")
    sys.exit(1)
except botocore.exceptions.ClientError as e:
    logger.error(f"Error initializing S3 client: {e}", exc_info=True)
    print(f"Error initializing S3 client: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error initializing S3 client: {e}", exc_info=True)
    print(f"Unexpected error initializing S3 client: {e}")
    sys.exit(1)

# --- Helper Functions ---
def construct_s3_url(bucket, region, s3_key):
    """Constructs a standard S3 URL."""
    encoded_key = urllib.parse.quote(s3_key, safe='/')
    return f'https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}'

def upload_file_if_needed(s3_client, local_path, bucket, s3_key, content_type, description):
    """
    Checks if a file exists on S3. If not, uploads it from local_path.
    Returns the S3 URL if successful (exists or uploaded), otherwise None.
    """
    s3_url = construct_s3_url(bucket, s3_client.meta.region_name, s3_key)
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        logger.info(f"  ✅ {description} already exists on S3: s3://{bucket}/{s3_key}")
        print(f"  ✅ {description} already exists on S3.")
        return s3_url
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == '404':
            logger.info(f"  ⏳ {description} not found on S3. Attempting upload from {local_path}...")
            print(f"  ⏳ {description} not found on S3. Uploading...")
            try:
                extra_args = {}
                if content_type:
                    extra_args['ContentType'] = content_type
                s3_client.upload_file(
                    Filename=local_path,
                    Bucket=bucket,
                    Key=s3_key,
                    ExtraArgs=extra_args
                )
                logger.info(f"  ⬆️ Successfully uploaded {description} to s3://{bucket}/{s3_key}")
                print(f"  ⬆️ Successfully uploaded {description}.")
                return s3_url
            except FileNotFoundError:
                logger.error(f"  ❌ Upload failed: Local {description} file not found at {local_path}")
                print(f"  ❌ ERROR: Upload failed - Local {description} file not found: {os.path.basename(local_path)}")
                return None
            except Exception as upload_e:
                logger.error(f"  ❌ Failed to upload {description}: {upload_e}", exc_info=True)
                print(f"  ❌ ERROR: Failed to upload {description}. Error: {upload_e}")
                return None
        else:
            logger.error(f"  ❌ Error checking S3 object {s3_key}: {e}")
            print(f"  ❌ Error checking S3 object {description}: {e}.")
            return None

def find_file_by_doc_id(doc_id, directory, extensions=None):
    """
    Find files in the given directory that start with the document ID.
    If extensions is provided, only files with those extensions will be considered.
    Returns the first matching file path or None if not found.
    """
    if not os.path.exists(directory):
        return None
    
    for filename in os.listdir(directory):
        # Check if the filename starts with the document ID
        if filename.startswith(doc_id):
            # If extensions is provided, check if the file has one of those extensions
            if extensions:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions:
                    return os.path.join(directory, filename)
            else:
                return os.path.join(directory, filename)
    
    return None

def find_doc_file_in_subdirs(doc_id, base_dir, extensions=None):
    """
    Search for a file with the given document ID in all subdirectories.
    Returns the full path to the first matching file or None if not found.
    """
    if not os.path.exists(base_dir):
        return None
    
    # First, search directly in the base directory
    direct_match = find_file_by_doc_id(doc_id, base_dir, extensions)
    if direct_match:
        return direct_match
    
    # Then search in all subdirectories
    for dirpath, _, _ in os.walk(base_dir):
        # Skip the base directory since we already checked it
        if dirpath == base_dir:
            continue
        
        file_path = find_file_by_doc_id(doc_id, dirpath, extensions)
        if file_path:
            return file_path
    
    return None

def get_document_type_from_json(json_path):
    """
    Extract the document type from a JSON file.
    Returns the document type as a string.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list) and len(data) > 0:
            # For array-structured JSONs like ISM Manual
            return data[0].get('type', 'Unknown')
        elif isinstance(data, dict):
            # For single object JSONs like Fleet Alert
            return data.get('type', 'Unknown')
        else:
            return 'Unknown'
    except Exception as e:
        logger.error(f"Error extracting document type from {json_path}: {e}")
        return 'Unknown'

def update_json_with_urls(json_path, pdf_url, original_url):
    """
    Update a JSON file with the PDF and original file URLs.
    Returns True if successful, False otherwise.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Track if any changes were made
        update_made = False
        
        # Handle different JSON structures
        if isinstance(data, list):
            # For array-structured JSONs like ISM Manual
            for item in data:
                if isinstance(item, dict):
                    item['documentLink'] = pdf_url
                    item['downloadLink'] = original_url
                    update_made = True
        elif isinstance(data, dict):
            # For single object JSONs like Fleet Alert
            data['documentLink'] = pdf_url
            data['downloadLink'] = original_url
            update_made = True
        
        if update_made:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        else:
            logger.warning(f"No updates made to {json_path}")
            return False
    except Exception as e:
        logger.error(f"Error updating JSON file {json_path}: {e}")
        return False

def get_content_type(file_path):
    """
    Determine the content type based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    content_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.ppt': 'application/vnd.ms-powerpoint'
    }
    return content_types.get(ext, 'application/octet-stream')

# --- Main Processing Function ---
def process_documents():
    """
    Main function to process all documents.
    """
    logger.info("Starting document processing")
    print("Starting document processing")
    
    # Initialize counters
    total_json_files = 0
    processed_json_files = 0
    updated_json_files = 0
    skipped_json_files = 0
    error_json_files = 0
    missing_pdf_files = []
    missing_original_files = []
    missing_both_files = []
    
    # Process each document type directory
    for doc_type in DOCUMENT_TYPES:
        doc_type_dir = os.path.join(JSON_BASE_DIR, doc_type)
        
        if not os.path.exists(doc_type_dir):
            logger.info(f"Directory not found for document type: {doc_type}")
            print(f"Directory not found for document type: {doc_type}")
            continue
        
        logger.info(f"Processing document type: {doc_type}")
        print(f"\nProcessing document type: {doc_type}")
        
        # Get all JSON files in the document type directory
        json_files = [f for f in os.listdir(doc_type_dir) if f.endswith(JSON_SUFFIX)]
        total_json_files += len(json_files)
        
        if not json_files:
            logger.info(f"No JSON files found in {doc_type_dir}")
            print(f"No JSON files found in {doc_type_dir}")
            continue
        
        logger.info(f"Found {len(json_files)} JSON files for {doc_type}")
        print(f"Found {len(json_files)} JSON files for {doc_type}")
        
        # Map document type to S3 folder
        s3_folder = DOCUMENT_TYPE_TO_S3_FOLDER.get(doc_type, doc_type.replace(" ", "_"))
        
        # Define S3 subfolder paths
        s3_pdf_folder = f"{S3_BASE_FOLDER}/{s3_folder}/{doc_type} Pdf"
        s3_original_folder = f"{S3_BASE_FOLDER}/{s3_folder}/{doc_type} original Format"
        
        # Process each JSON file
        for json_file in tqdm(json_files, desc=f"Processing {doc_type}"):
            processed_json_files += 1
            json_path = os.path.join(doc_type_dir, json_file)
            
            # Extract document ID
            match = re.match(r'^(\d+)_', json_file)
            if not match:
                logger.warning(f"Could not extract document ID from {json_file}, skipping")
                print(f"  ⚠️ Could not extract document ID from {json_file}, skipping")
                skipped_json_files += 1
                continue
            
            doc_id = match.group(1)
            logger.info(f"Processing document ID: {doc_id}")
            print(f"\n  📄 Processing document ID: {doc_id}")
            
            # Find PDF file
            pdf_path = find_doc_file_in_subdirs(doc_id, SOURCE_FILES_DIR, ['.pdf'])
            pdf_found = pdf_path is not None
            
            if pdf_found:
                pdf_filename = os.path.basename(pdf_path)
                logger.info(f"Found PDF file: {pdf_filename}")
                print(f"  ✅ Found PDF file: {pdf_filename}")
            else:
                logger.warning(f"PDF file not found for document ID: {doc_id}")
                print(f"  ⚠️ PDF file not found for document ID: {doc_id}")
                missing_pdf_files.append(json_file)
            
            # Find original file (non-PDF)
            original_path = None
            original_found = False
            for ext in ALLOWED_ORIGINAL_EXTENSIONS:
                original_path = find_doc_file_in_subdirs(doc_id, SOURCE_FILES_DIR, [ext])
                if original_path:
                    original_filename = os.path.basename(original_path)
                    logger.info(f"Found original file: {original_filename}")
                    print(f"  ✅ Found original file: {original_filename}")
                    original_found = True
                    break
            
            # If no original file found, use PDF as original
            if not original_found:
                if pdf_found:
                    original_path = pdf_path
                    original_filename = pdf_filename
                    logger.info(f"Using PDF as original file: {original_filename}")
                    print(f"  ℹ️ Using PDF as original file")
                else:
                    logger.warning(f"No original file found for document ID: {doc_id}")
                    print(f"  ⚠️ No original file found for document ID: {doc_id}")
                    missing_original_files.append(json_file)
                    missing_both_files.append(json_file)
                    skipped_json_files += 1
                    continue
            
            # Upload PDF to S3
            pdf_s3_url = None
            if pdf_found:
                pdf_s3_key = f"{s3_pdf_folder}/{pdf_filename}"
                pdf_s3_url = upload_file_if_needed(
                    s3_client, 
                    pdf_path, 
                    BUCKET_NAME, 
                    pdf_s3_key, 
                    'application/pdf', 
                    'PDF'
                )
                if not pdf_s3_url:
                    logger.error(f"Failed to upload or get URL for PDF: {pdf_filename}")
                    print(f"  ❌ Failed to upload or get URL for PDF")
            
            # Upload original file to S3
            original_s3_url = None
            if original_path:
                original_s3_key = f"{s3_original_folder}/{os.path.basename(original_path)}"
                original_content_type = get_content_type(original_path)
                original_s3_url = upload_file_if_needed(
                    s3_client, 
                    original_path, 
                    BUCKET_NAME, 
                    original_s3_key, 
                    original_content_type, 
                    'Original'
                )
                if not original_s3_url:
                    logger.error(f"Failed to upload or get URL for original file: {os.path.basename(original_path)}")
                    print(f"  ❌ Failed to upload or get URL for original file")
            
            # Update JSON file with URLs
            if pdf_s3_url and original_s3_url:
                if update_json_with_urls(json_path, pdf_s3_url, original_s3_url):
                    logger.info(f"Successfully updated JSON file: {json_file}")
                    print(f"  ✅ Successfully updated JSON file")
                    updated_json_files += 1
                else:
                    logger.error(f"Failed to update JSON file: {json_file}")
                    print(f"  ❌ Failed to update JSON file")
                    error_json_files += 1
            else:
                logger.warning(f"Skipping JSON update for {json_file} (missing URLs)")
                print(f"  ⚠️ Skipping JSON update (missing URLs)")
                skipped_json_files += 1
    
    # Print summary
    logger.info("\n--- Processing Summary ---")
    logger.info(f"Total JSON files processed: {processed_json_files}/{total_json_files}")
    logger.info(f"JSON files successfully updated: {updated_json_files}")
    logger.info(f"JSON files skipped: {skipped_json_files}")
    logger.info(f"JSON files with errors: {error_json_files}")
    logger.info(f"Missing PDF files: {len(missing_pdf_files)}")
    logger.info(f"Missing original files: {len(missing_original_files)}")
    logger.info(f"Missing both PDF and original files: {len(missing_both_files)}")
    
    print("\n=== Processing Summary ===")
    print(f"Total JSON files processed: {processed_json_files}/{total_json_files}")
    print(f"JSON files successfully updated: {updated_json_files}")
    print(f"JSON files skipped: {skipped_json_files}")
    print(f"JSON files with errors: {error_json_files}")
    print(f"Missing PDF files: {len(missing_pdf_files)}")
    print(f"Missing original files: {len(missing_original_files)}")
    print(f"Missing both PDF and original files: {len(missing_both_files)}")
    
    return {
        'success': True,
        'total_json_files': total_json_files,
        'processed_json_files': processed_json_files,
        'updated_json_files': updated_json_files,
        'skipped_json_files': skipped_json_files,
        'error_json_files': error_json_files
    }

# --- Main Execution ---
if __name__ == "__main__":
    try:
        result = process_documents()
        print("\nDocument S3 upload and JSON update completed successfully!")
        logger.info("Document S3 upload and JSON update completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1) 