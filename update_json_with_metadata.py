import os
import json
import re
import logging
from pathlib import Path
import csv
import time
import argparse

# --- Setup Logging ---
def setup_logging():
    log_filename = f"update_json_log_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

# --- Parse Command Line Arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Update JSON files with metadata from a CSV file")
    
    parser.add_argument(
        "--json-dir", 
        type=str,
        default=os.getenv("JSON_OUTPUT_DIR", "parsed_json_results"),
        help="Directory containing JSON files to update (env: JSON_OUTPUT_DIR)"
    )
    
    parser.add_argument(
        "--csv-file",
        type=str,
        default=os.getenv("CSV_FILE_PATH", "docmap_final_processed.csv"),
        help="Path to CSV file with document metadata (env: CSV_FILE_PATH)"
    )
    
    parser.add_argument(
        "--doc-no-col",
        type=str,
        default="Doc. No",
        help="CSV column name for document number"
    )
    
    parser.add_argument(
        "--doc-title-col",
        type=str,
        default="Document title",
        help="CSV column name for document title"
    )
    
    parser.add_argument(
        "--doc-type-col",
        type=str,
        default="Type",
        help="CSV column name for document type"
    )
    
    parser.add_argument(
        "--doc-id-col",
        type=str,
        default="DocID",
        help="CSV column name for document ID"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Increase output verbosity"
    )
    
    parser.add_argument(
        "--keep-keywords",
        action="store_true",
        help="Keep keywords field in JSON files if present (default is to remove keywords)"
    )
    
    parser.add_argument(
        "--target-file",
        type=str,
        help="Process a specific JSON file instead of all files in directory"
    )
    
    return parser.parse_args()

# --- Function to Load Data from CSV ---
def load_doc_map_from_csv(csv_path, doc_no_col, doc_title_col, doc_type_col, doc_id_col):
    """
    Reads the CSV file and creates a dictionary mapping DocID to Doc_No, Document Title, and Type.
    """
    doc_map = {}
    logging.info(f"Attempting to load document map from CSV: {csv_path}")
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            header = reader.fieldnames
            if not header:
                logging.error("CSV file is empty or has no header row.")
                return None
            
            required_cols = [doc_no_col, doc_title_col, doc_type_col, doc_id_col]
            missing_cols = [col for col in required_cols if col not in header]
            if missing_cols:
                logging.error(f"CSV file is missing required columns: {', '.join(missing_cols)}. Required: {required_cols}")
                return None

            # Process rows
            row_count = 0
            skipped_rows = 0
            for row in reader:
                row_count += 1
                doc_id = row.get(doc_id_col, '').strip()
                doc_no = row.get(doc_no_col, '').strip()
                doc_title = row.get(doc_title_col, '').strip()
                doc_type = row.get(doc_type_col, '').strip()

                # Validate DocID
                if doc_id:
                    doc_map[doc_id] = {
                        "docNo": doc_no if doc_no else 'N/A',
                        "documentTitle": doc_title if doc_title else 'N/A',
                        "type": doc_type if doc_type else 'N/A',
                        "docId": doc_id
                    }
                else:
                    logging.warning(f"Skipping row {row_count} in CSV: Missing DocID.")
                    skipped_rows += 1

        logging.info(f"Successfully loaded data for {len(doc_map)} unique DocIDs from {row_count} rows (skipped {skipped_rows} rows).")
        return doc_map

    except FileNotFoundError:
        logging.error(f"CSV file not found at: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}", exc_info=True)
        return None

# --- Process JSON Files in Directory Structure ---
def process_directory(directory, doc_map, verbose=False, keep_keywords=False, target_file=None):
    """
    Recursively processes all JSON files in the given directory and its subdirectories.
    If target_file is specified, only process that file.
    """
    processed_files = 0
    skipped_files = 0
    
    try:
        # Convert to Path object if string is provided
        if isinstance(directory, str):
            directory = Path(directory)
            
        logging.info(f"Processing directory: {directory}")
        
        # If target_file is specified, only process that file
        if target_file:
            target_path = Path(target_file)
            # If target_file is an absolute path, use it directly
            if not target_path.is_absolute():
                # Otherwise, assume it's relative to the directory
                target_path = directory / target_path
                
            if target_path.exists() and target_path.suffix.lower() == '.json':
                logging.info(f"Processing specific target file: {target_path}")
                if process_json_file(target_path, doc_map, verbose, keep_keywords):
                    processed_files += 1
                    logging.info(f"Successfully processed target file: {target_path}")
                else:
                    skipped_files += 1
                    logging.warning(f"Failed to process target file: {target_path}")
            else:
                logging.error(f"Target file not found or not a JSON file: {target_path}")
                skipped_files += 1
                
            return processed_files, skipped_files
        
        # Process all subdirectories and files
        for item in directory.iterdir():
            if item.is_dir():
                sub_processed, sub_skipped = process_directory(item, doc_map, verbose, keep_keywords)
                processed_files += sub_processed
                skipped_files += sub_skipped
            elif item.is_file() and item.suffix.lower() == '.json':
                # Process JSON file
                if process_json_file(item, doc_map, verbose, keep_keywords):
                    processed_files += 1
                else:
                    skipped_files += 1
                    
        return processed_files, skipped_files
        
    except Exception as e:
        logging.error(f"Error processing directory {directory}: {e}", exc_info=True)
        return processed_files, skipped_files

# --- Process Individual JSON File ---
def process_json_file(json_path, doc_map, verbose=False, keep_keywords=False):
    """
    Updates a single JSON file with metadata from the doc_map.
    Returns True if successful, False otherwise.
    """
    filename = json_path.name
    logging.info(f"Processing file: {filename}")
    
    try:
        # Extract DocID from filename
        match = re.search(r'(\d+)_', filename)
        if not match:
            logging.warning(f"  Skipping file {filename}: Could not extract DocID from filename.")
            return False
            
        doc_id = match.group(1)
        if verbose:
            logging.info(f"  Extracted DocID from filename: {doc_id}")
        
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logging.warning(f"  Skipping file {filename}: File is empty.")
                return False
                
            data = json.loads(content)
        
        # Track if any changes were made to the file
        changes_made = False
        
        # Look up data in the CSV map
        csv_data = doc_map.get(doc_id)
        process_full_metadata = True
        
        if not csv_data:
            logging.warning(f"  DocID '{doc_id}' not found in the provided CSV map. Will only process documentHeader and embText fields.")
            process_full_metadata = False
            # Create a minimal csv_data structure with empty values
            csv_data = {
                "docNo": 'N/A',
                "documentTitle": '',
                "type": '',
                "docId": doc_id
            }
        
        # Process the data based on its structure
        if isinstance(data, dict):
            # Process a single dictionary object
            if verbose:
                logging.info(f"  Processing single dictionary object")
                
            # Always remove keywords field unless explicitly told to keep it
            if 'keywords' in data and not keep_keywords:
                logging.info(f"  Removing 'keywords' field from {filename}")
                del data['keywords']
                changes_made = True
                
            if process_full_metadata:
                if update_json_item(data, csv_data, verbose, keep_keywords):
                    changes_made = True
            else:
                if add_document_header_and_embed_text(data, verbose, keep_keywords):
                    changes_made = True
                    
            # Verify the fields were added and retained
            if verbose:
                logging.info(f"  Single object fields: {', '.join(data.keys())}")
            if "documentHeader" not in data or "embText" not in data:
                logging.warning(f"  Some fields missing after update: documentHeader={'documentHeader' in data}, embText={'embText' in data}")
        elif isinstance(data, list):
            # Process a list of items
            if verbose:
                logging.info(f"  Processing list of {len(data)} items")
            items_updated = 0
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # Always remove keywords field unless explicitly told to keep it
                    if 'keywords' in item and not keep_keywords:
                        logging.info(f"  Removing 'keywords' field from item {i} in {filename}")
                        del item['keywords']
                        changes_made = True
                        
                    if process_full_metadata:
                        if update_json_item(item, csv_data, verbose, keep_keywords):
                            changes_made = True
                            items_updated += 1
                    else:
                        if add_document_header_and_embed_text(item, verbose, keep_keywords):
                            changes_made = True
                            items_updated += 1
                            
                    # Verify for a sample of items (first, last, and middle)
                    if verbose and (i == 0 or i == len(data)-1 or i == len(data)//2):
                        logging.info(f"  Item {i} fields: {', '.join(item.keys())}")
                        if "documentHeader" not in item or "embText" not in item:
                            logging.warning(f"  Item {i}: Some fields missing after update: documentHeader={'documentHeader' in item}, embText={'embText' in item}")
            if verbose:
                logging.info(f"  Updated {items_updated} of {len(data)} items")
        else:
            logging.warning(f"  Skipping file {filename}: JSON content is neither a dictionary nor a list.")
            return False
            
        # Write the updated JSON data back to the file if changes were made
        if changes_made:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            logging.info(f"  Successfully updated metadata for: {filename}")
            return True
        else:
            logging.info(f"  No changes needed for: {filename}")
            return True
            
    except json.JSONDecodeError as e:
        logging.error(f"  Error decoding JSON from file: {filename}. Error: {e}")
        return False
    except Exception as e:
        logging.error(f"  An unexpected error occurred while processing {filename}: {e}", exc_info=True)
        return False

def update_json_item(item, csv_data, verbose=False, keep_keywords=False):
    """
    Updates a single JSON dictionary with metadata fields.
    Returns True if changes were made, False otherwise.
    """
    try:
        changes_made = False
        
        # Always remove keywords field unless explicitly told to keep it
        if 'keywords' in item and not keep_keywords:
            del item['keywords']
            changes_made = True
            if verbose:
                logging.info(f"  Removed 'keywords' field from item")
        
        # Add metadata fields from CSV
        item["docNo"] = csv_data["docNo"]
        item["documentTitle"] = csv_data["documentTitle"]
        item["type"] = csv_data["type"]
        item["docId"] = csv_data["docId"]
        
        # Get values for additional fields, using empty strings for missing values
        doc_name = item.get('documentName', '')
        doc_title = item.get("documentTitle", '')
        section = item.get('section', '')
        chapter = item.get('chapter', '')
        doc_type = item.get("type", '')
        original_text = item.get('originalText', '')
        
        # Convert page to pageNumber if needed
        page = item.get('page', '')
        if 'page' in item:
            item['pageNumber'] = item['page']
            if 'pageNumber' not in item:  # Only delete if we successfully copied
                del item['page']
            changes_made = True
        elif 'pageNumber' not in item:
            item['pageNumber'] = ''
            changes_made = True
        
        if verbose:
            logging.info(f"  Building header with: doc_name={doc_name}, doc_title={doc_title}, section={section}, chapter={chapter}, doc_type={doc_type}, pageNumber={item['pageNumber']}")
        
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
            logging.warning(f"  Created minimal header with only title/type as no other header fields were available")
        
        document_header = " | ".join(header_parts)
        prev_header = item.get("documentHeader", "")
        if document_header != prev_header:
            item["documentHeader"] = document_header
            changes_made = True
            if verbose:
                logging.info(f"  Created documentHeader: {document_header}")
        
        # Create embText field combining available fields plus original text
        embed_parts = header_parts.copy()
        if original_text: embed_parts.append(f"Original text: {original_text}")
        
        embed_text = " | ".join(embed_parts)
        prev_embed_text = item.get("embText", "")
        if embed_text != prev_embed_text:
            item["embText"] = embed_text
            changes_made = True
            if verbose:
                logging.info(f"  Created embText field (length: {len(embed_text)})")
        
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
            "docNo": item.get("docNo", ""),
            "docId": item.get("docId", ""),
        }
        
        # For non-ISM and non-Office Manual documents, copy originalText to shortSummary
        is_ism_or_office = doc_type in ["ISM Manual", "Office Manual"]
        if not is_ism_or_office and original_text and not item.get('shortSummary'):
            item['shortSummary'] = original_text
            changes_made = True
            if verbose:
                logging.info(f"  Copied originalText to shortSummary for non-ISM/Office document")
        
        # Add any missing fields
        for field, default_value in required_fields.items():
            if field not in item:
                item[field] = default_value
                changes_made = True
                if verbose:
                    logging.info(f"  Added missing field '{field}' with default value")
        
        # Verify fields were added
        if "documentHeader" not in item:
            logging.error(f"  Failed to add documentHeader to item despite no exceptions")
        if "embText" not in item:
            logging.error(f"  Failed to add embText to item despite no exceptions")
            
        return changes_made
    except Exception as e:
        logging.error(f"  Error updating JSON item: {e}", exc_info=True)
        return False

def add_document_header_and_embed_text(item, verbose=False, keep_keywords=False):
    """
    Adds only documentHeader and embText fields to a JSON item.
    Returns True if changes were made, False otherwise.
    """
    try:
        changes_made = False
        
        # Always remove keywords field unless explicitly told to keep it
        if 'keywords' in item and not keep_keywords:
            del item['keywords']
            changes_made = True
            if verbose:
                logging.info(f"  Removed 'keywords' field from item")
        
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
            changes_made = True
        elif 'pageNumber' not in item:
            item['pageNumber'] = ''
            changes_made = True
        
        if verbose:
            logging.info(f"  Building header with: doc_name={doc_name}, doc_title={doc_title}, section={section}, chapter={chapter}, doc_type={doc_type}, pageNumber={item['pageNumber']}")
        
        # Create documentHeader field combining available fields
        header_parts = []
        if doc_name: header_parts.append(f"Document name: {doc_name}")
        if doc_title: header_parts.append(f"Document title: {doc_title}")
        if section: header_parts.append(f"Section name: {section}")
        if chapter: header_parts.append(f"Chapter name: {chapter}")
        if doc_type: header_parts.append(f"Type: {doc_type}")
        if item['pageNumber']: header_parts.append(f"Page number: {item['pageNumber']}")
        
        # If header_parts is empty, try the documentHeader if it exists already
        if not header_parts and "documentHeader" in item:
            if verbose:
                logging.info(f"  Using existing documentHeader: {item['documentHeader']}")
            document_header = item["documentHeader"]
        else:
            document_header = " | ".join(header_parts)
            if document_header != item.get("documentHeader", ""):
                item["documentHeader"] = document_header
                changes_made = True
                if verbose:
                    logging.info(f"  Created documentHeader: {document_header}")
        
        # Create embText field combining available fields plus original text
        embed_parts = []
        if document_header: 
            embed_parts.append(document_header)
        if original_text: 
            embed_parts.append(f"Original text: {original_text}")
        
        embed_text = " | ".join(embed_parts)
        if embed_text != item.get("embText", ""):
            item["embText"] = embed_text
            changes_made = True
            if verbose:
                logging.info(f"  Created embText field (length: {len(embed_text)})")
        
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
        
        # For non-manual documents, make sure originalText is used
        is_manual_type = doc_type in ["ISM Manual", "Office Manual", "Policies", "Sample Document Type"]
        if not is_manual_type and content and not item.get('originalText'):
            item['originalText'] = content[:500]  # Use truncated content as original text
            changes_made = True
            if verbose:
                logging.info(f"  Copied originalText to shortSummary for non-ISM/Office document")
        
        # Add any missing fields
        for field, default_value in required_fields.items():
            if field not in item:
                item[field] = default_value
                changes_made = True
                if verbose:
                    logging.info(f"  Added missing field '{field}' with default value")
        
        return changes_made
    except Exception as e:
        logging.error(f"  Error adding documentHeader and embText: {e}", exc_info=True)
        return False

# --- Main Execution ---
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    log_filename = setup_logging()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled")
    
    start_time = time.time()
    
    # Log the configuration
    logging.info(f"Configuration:")
    logging.info(f"- JSON directory: {args.json_dir}")
    logging.info(f"- CSV file: {args.csv_file}")
    logging.info(f"- CSV columns: {args.doc_no_col}, {args.doc_title_col}, {args.doc_type_col}, {args.doc_id_col}")
    if args.keep_keywords:
        logging.info(f"- Keep keywords field: Yes")
    else:
        logging.info(f"- Remove keywords field: Yes (default)")
    if args.target_file:
        logging.info(f"- Target file: {args.target_file}")
    
    # 1. Load the mapping from CSV
    document_map = load_doc_map_from_csv(
        args.csv_file, 
        args.doc_no_col, 
        args.doc_title_col, 
        args.doc_type_col, 
        args.doc_id_col
    )
    
    # 2. Process JSON files only if the map was loaded successfully
    if document_map is not None:
        processed_files, skipped_files = process_directory(
            args.json_dir, 
            document_map, 
            args.verbose, 
            args.keep_keywords, 
            args.target_file
        )
        logging.info("--- Processing Summary ---")
        logging.info(f"Total files processed: {processed_files + skipped_files}")
        logging.info(f"Files successfully updated: {processed_files}")
        logging.info(f"Files skipped: {skipped_files}")
    else:
        logging.error("Failed to load document map from CSV. Aborting JSON file processing.")
    
    end_time = time.time()
    logging.info(f"Script finished in {end_time - start_time:.2f} seconds.")
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
    print(f"Check log file '{log_filename}' for details.")
    
    return True

if __name__ == "__main__":
    main() 