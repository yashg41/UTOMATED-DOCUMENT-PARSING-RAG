#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document to PDF Converter

This module uses Adobe PDF Services API to convert various document types to PDF.
It retains the original files and creates PDF versions alongside them.
"""

import os
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adobe SDK Imports
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.create_pdf_job import CreatePDFJob
from adobe.pdfservices.operation.pdfjobs.result.create_pdf_result import CreatePDFResult

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFConverter:
    """Class to handle document conversion to PDF using Adobe API."""
    
    def __init__(self, client_id, client_secret, max_workers=3):
        """
        Initialize PDF Converter with Adobe credentials.
        
        Args:
            client_id (str): Adobe API client ID
            client_secret (str): Adobe API client secret
            max_workers (int): Maximum number of concurrent conversions
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.max_workers = max_workers
        self.credentials = None
        
        # Initialize credentials
        try:
            self.credentials = ServicePrincipalCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            logger.info("Adobe Credentials created successfully.")
        except Exception as e:
            logger.error(f"Error initializing Adobe credentials: {e}")
            raise
    
    def get_mime_type(self, file_path):
        """
        Determine the Adobe PDF Services SDK MIME type based on file extension.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str or None: MIME type for the file or None if unsupported
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        
        mime_types = {
            '.docx': PDFServicesMediaType.DOCX,
            '.doc': PDFServicesMediaType.DOC,
            '.xlsx': PDFServicesMediaType.XLSX,
            '.xls': PDFServicesMediaType.XLS,
            '.pptx': PDFServicesMediaType.PPTX,
            '.ppt': PDFServicesMediaType.PPT,
            '.rtf': PDFServicesMediaType.RTF,
            '.txt': PDFServicesMediaType.TXT
        }
        
        return mime_types.get(extension)
    
    def is_approved_category(self, file_path):
        """
        Check if the file belongs to one of the approved document categories.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if file belongs to an approved category, False otherwise
        """
        # List of approved document categories
        approved_categories = [
            'ra database',
            'forms and checklists',
            'standard eform',
            'fleet alert',
            'office manuals',
            'ism manuals',
            'office forms',
            'policies',
            'sample document type'
        ]
        
        # Convert file path to lowercase for case-insensitive comparison
        file_path_lower = file_path.lower()
        
        # Check if any of the approved categories appears in the file path
        for category in approved_categories:
            if category in file_path_lower:
                logger.info(f"File {os.path.basename(file_path)} belongs to category: {category}")
                return True
                
        logger.info(f"File {os.path.basename(file_path)} does not belong to any approved category - skipping")
        return False
    
    def convert_file(self, file_path):
        """
        Convert a single file to PDF and save it with the same name but .pdf extension.
        
        Args:
            file_path (str): Path to the file to convert
            
        Returns:
            dict: Result information with status, input file, and output file
        """
        thread_name = threading.current_thread().name
        input_filename = os.path.basename(file_path)
        logger.info(f"[{thread_name}] Starting conversion for: {input_filename}")
        
        try:
            # Create PDFServices instance
            pdf_services = PDFServices(credentials=self.credentials)
            
            # Determine MIME Type
            mime_type = self.get_mime_type(file_path)
            if not mime_type:
                raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}")
                
            # Read input file
            with open(file_path, 'rb') as file:
                input_stream = file.read()
                
            # Upload input asset
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=mime_type)
            
            # Create and submit conversion job
            create_pdf_job = CreatePDFJob(input_asset)
            location = pdf_services.submit(create_pdf_job)
            logger.info(f"[{thread_name}] Job submitted for {input_filename}")
            
            # Get job result
            pdf_services_response = pdf_services.get_job_result(location, CreatePDFResult)
            
            # Get and save PDF
            result_asset = pdf_services_response.get_result().get_asset()
            stream_asset = pdf_services.get_content(result_asset)
            
            base_name = os.path.splitext(input_filename)[0]
            output_dir = os.path.dirname(file_path)
            output_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
            
            with open(output_pdf_path, "wb") as file:
                file.write(stream_asset.get_input_stream())
            logger.info(f"[{thread_name}] Successfully saved PDF: {os.path.basename(output_pdf_path)}")
            
            return {
                'status': 'success',
                'input': input_filename,
                'output': os.path.basename(output_pdf_path),
                'output_path': output_pdf_path
            }
            
        except Exception as e:
            logger.error(f"[{thread_name}] ❌ Error converting {input_filename}: {e}")
            return {
                'status': 'error',
                'input': input_filename,
                'message': str(e)
            }
    
    def find_files_to_convert(self, directory):
        """
        Find all files that need to be converted to PDF in the given directory.
        
        Args:
            directory (str): Root directory to search for files
            
        Returns:
            list: Paths of files to convert
        """
        files_to_convert = []
        
        for dirpath, _, filenames in os.walk(directory):
            logger.info(f"Scanning directory: {dirpath}")
            
            # Get existing PDF basenames in this directory
            existing_pdfs = set()
            for filename in filenames:
                if filename.lower().endswith('.pdf'):
                    base_name = os.path.splitext(filename)[0].lower()
                    existing_pdfs.add(base_name)
            
            # Find convertible files
            for filename in filenames:
                if filename.startswith('.'):
                    continue  # Skip hidden files
                    
                file_path = os.path.join(dirpath, filename)
                if not os.path.isfile(file_path):
                    continue
                    
                # Skip existing PDFs
                if filename.lower().endswith('.pdf'):
                    continue
                    
                # Skip if PDF version already exists
                base_name = os.path.splitext(filename)[0].lower()
                if base_name in existing_pdfs:
                    logger.info(f"Skipping {filename} - corresponding PDF already exists")
                    continue
                
                # Check if file belongs to an approved document category
                if not self.is_approved_category(file_path):
                    continue
                    
                # Check if file type is convertible
                if self.get_mime_type(file_path):
                    files_to_convert.append(file_path)
                    logger.debug(f"Added for conversion: {file_path}")
                else:
                    logger.info(f"Unsupported file type: {filename}")
        
        return files_to_convert
    
    def batch_convert_directory(self, input_directory):
        """
        Convert all supported files in a directory tree to PDF.
        
        Args:
            input_directory (str): Root directory to process
            
        Returns:
            dict: Summary of conversion results
        """
        start_time = time.time()
        logger.info(f"--- Starting Recursive Batch Conversion Process ---")
        logger.info(f"Root Directory: {input_directory}")
        
        if not os.path.isdir(input_directory):
            logger.error(f"❌ ERROR: Input directory not found: '{input_directory}'")
            return {
                'success': False,
                'error': f"Directory not found: {input_directory}"
            }
        
        try:
            # Find all files to convert
            files_to_convert = self.find_files_to_convert(input_directory)
            logger.info(f"Found {len(files_to_convert)} file(s) requiring conversion.")
            
            # Initialize counters
            success_count = 0
            error_count = 0
            converted_files = []
            failed_files = []
            
            # Process files in parallel
            if files_to_convert:
                with ThreadPoolExecutor(
                    max_workers=self.max_workers, 
                    thread_name_prefix='PDFWorker'
                ) as executor:
                    logger.info(f"Submitting {len(files_to_convert)} tasks to {self.max_workers} workers...")
                    
                    # Submit conversion tasks
                    futures = [
                        executor.submit(self.convert_file, file_path) 
                        for file_path in files_to_convert
                    ]
                    
                    # Process results
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result['status'] == 'success':
                                success_count += 1
                                converted_files.append(result)
                                logger.info(f"✅ Converted: {result['input']} -> {result['output']}")
                            else:
                                error_count += 1
                                failed_files.append(result)
                                logger.error(f"❌ Failed: {result['input']} - {result.get('message', 'Unknown error')}")
                        except Exception as exc:
                            error_count += 1
                            logger.error(f"❌ Task failed: {exc}")
            else:
                logger.info("No files found requiring conversion.")
            
            # Final Summary
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"\n--- Conversion Summary ---")
            logger.info(f"Total files processed: {len(files_to_convert)}")
            logger.info(f"Successfully converted: {success_count}")
            logger.info(f"Failed conversions: {error_count}")
            logger.info(f"Time taken: {duration:.2f} seconds")
            
            return {
                'success': True,
                'total_files': len(files_to_convert),
                'success_count': success_count,
                'error_count': error_count,
                'duration_seconds': duration,
                'converted_files': converted_files,
                'failed_files': failed_files
            }
            
        except Exception as e:
            logger.critical(f"❌ CRITICAL ERROR: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

def convert_to_pdf(client_id, client_secret, input_directory, max_workers=3):
    """
    Main function to convert documents to PDF.
    
    Args:
        client_id (str): Adobe API client ID
        client_secret (str): Adobe API client secret
        input_directory (str): Directory containing files to convert
        max_workers (int): Maximum number of concurrent conversions
        
    Returns:
        dict: Summary of conversion results
    """
    try:
        converter = PDFConverter(client_id, client_secret, max_workers)
        return converter.batch_convert_directory(input_directory)
    except Exception as e:
        logger.critical(f"Failed to initialize PDF converter: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Configuration from environment variables
    ADOBE_CLIENT_ID = os.getenv("ADOBE_CLIENT_ID")
    ADOBE_CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET")
    INPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "downloaded_data")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
    
    # Check if credentials are available
    if not ADOBE_CLIENT_ID or not ADOBE_CLIENT_SECRET:
        logger.error("❌ Error: ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET must be set in .env file")
        sys.exit(1)
    
    # Execute conversion
    result = convert_to_pdf(ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET, INPUT_DIRECTORY, MAX_WORKERS)
    
    if result['success']:
        print(f"\nConversion completed successfully!")
        print(f"Converted {result['success_count']} of {result['total_files']} files")
    else:
        print(f"\nConversion failed: {result.get('error', 'Unknown error')}") 