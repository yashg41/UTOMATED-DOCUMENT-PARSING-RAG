#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MongoDB to Typesense Sync Script

This script synchronizes documents from MongoDB to Typesense:
1. Connects to both MongoDB and Typesense
2. Fetches existing document IDs from Typesense
3. Retrieves only new documents from MongoDB
4. Transforms MongoDB documents to match Typesense schema
5. Uploads documents in batches with proper error handling
"""

import pymongo
from pymongo import MongoClient
import typesense
import os
import math
import time
import json
from bson import ObjectId  # To handle MongoDB ObjectIds
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mongodb_to_typesense.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# MongoDB Configuration
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "")

# Typesense Configuration
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT", "443"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL", "https")
TYPESENSE_COLLECTION_NAME = os.getenv("TYPESENSE_COLLECTION_NAME", "")

# Indexing Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
IMPORT_ACTION = os.getenv("IMPORT_ACTION", "upsert")
CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "60"))

def prepare_document_for_typesense(mongo_doc):
    """
    Transforms a MongoDB document into the format expected by the Typesense schema.
    - Converts MongoDB '_id' (ObjectId) to Typesense 'id' (string).
    - Handles specific type conversions (pageNumber, chunkNo, context_list_ids).
    - Validates required fields based on the confirmed schema.
    - Handles potential None values for optional fields.
    """
    ts_doc = {}

    # --- ID Handling: Map MongoDB '_id' to Typesense 'id' ---
    if '_id' in mongo_doc and isinstance(mongo_doc['_id'], ObjectId):
        # This 'id' field will be used by Typesense as the document identifier
        ts_doc['id'] = str(mongo_doc['_id'])
    elif 'id' in mongo_doc:  # Less common case if 'id' already exists in Mongo source
        ts_doc['id'] = str(mongo_doc['id'])
    else:
        # Typesense requires an 'id' field for every document.
        logger.warning(f"Warning: Document missing '_id' or 'id'. Skipping: {mongo_doc.get('_id', 'N/A')}")
        return None
    # --- End ID Handling ---

    # --- Define Required Fields (based on your confirmed Typesense schema) ---
    # Fields marked "optional": false in your schema
    required_fields = [
        'docId',
        'docNo',
        'documentTitle',
        'documentLink',
        'type',
        'embText'
    ]

    # Iterate through other fields in the MongoDB document
    for key, value in mongo_doc.items():
        if key == '_id':  # Skip the original _id field, already handled as 'id'
            continue

        # --- Specific Field Transformations based on Schema ---
        if key == 'pageNumber':
            # Schema type: string, optional: true
            ts_doc[key] = str(value) if value is not None else None  # Keep None or convert to string "" if preferred
        elif key == 'chunkNo':
            # Schema type: int32, optional: true
            try:
                ts_doc[key] = int(value) if value is not None else None  # Allow None if source is None
            except (ValueError, TypeError):
                logger.warning(f"Warning: Could not convert chunkNo '{value}' (type: {type(value)}) to int for doc id {ts_doc['id']}. Setting to None.")
                ts_doc[key] = None
        elif key == 'context_list_ids':
            # Schema type: string[], optional: true
            # Mongo source: list of {"$oid": "..."} which Pymongo likely converts to ObjectId objects
            if isinstance(value, list):
                transformed_list = []
                valid = True
                for item in value:
                    # --- START REVISED LOGIC ---
                    # Check if item is an ObjectId object provided by PyMongo
                    if isinstance(item, ObjectId):
                        transformed_list.append(str(item))  # Convert ObjectId to string
                    # --- END REVISED LOGIC ---
                    elif item is None:  # Handle potential None values within the list if needed
                        continue  # Skip None items
                    # Fallback/Error Handling for unexpected types
                    elif isinstance(item, str):
                        # If for some reason raw strings ARE present in some docs
                        logger.info(f"Note: Found raw string '{item}' in context_list_ids for doc id {ts_doc['id']}. Adding as is.")
                        transformed_list.append(item)
                    elif isinstance(item, dict) and '$oid' in item and isinstance(item.get('$oid'), str):
                        # Fallback if PyMongo didn't convert it for some reason
                        logger.info(f"Note: Found dict format item {item} in context_list_ids for doc id {ts_doc['id']}. Extracting $oid.")
                        transformed_list.append(item['$oid'])
                    else:
                        # Log genuinely unexpected format within the list
                        logger.warning(f"Warning: Invalid item type in context_list_ids for doc id {ts_doc['id']}: Type={type(item)}, Value={item}. Skipping field for this doc.")
                        valid = False
                        break  # Stop processing this field for this doc
                if valid:
                    ts_doc[key] = transformed_list if transformed_list else None  # Assign list or None if empty/invalid
                else:
                    ts_doc[key] = None  # Set field to None if invalid items were found
            elif value is None:
                ts_doc[key] = None  # Keep None if source field is None
            else:
                # Log if the field is not a list or None
                logger.warning(f"Warning: Unexpected type for context_list_ids field for doc id {ts_doc['id']}: {type(value)}. Skipping field.")
                ts_doc[key] = None
        # --- End Specific Field Transformations ---

        # Add other fields directly, assuming names match between Mongo and Typesense
        else:
            # Add the field to the Typesense document
            # Typesense handles None for fields marked optional: true
            ts_doc[key] = value

    # *** Important Check for Required Fields ***
    missing_or_invalid_required = False
    for req_field in required_fields:
        # Check if the required field is missing or is None
        if req_field not in ts_doc or ts_doc[req_field] is None:
            # You might allow empty string "" for required string fields if acceptable
            if isinstance(ts_doc.get(req_field, None), str) and ts_doc.get(req_field) == "":
                # Allowing empty string for required string field
                continue

            logger.error(f"Error: Required field '{req_field}' is missing or None for doc id {ts_doc['id']}.")
            missing_or_invalid_required = True
            break  # No need to check further required fields for this doc

    if missing_or_invalid_required:
        return None  # Skip this document

    return ts_doc

def sync_mongodb_to_typesense():
    """
    Main function to synchronize MongoDB documents to Typesense.
    Fetches new documents from MongoDB and indexes them in Typesense.
    
    Returns:
        dict: Result summary with success and statistics
    """
    mongo_client = None
    typesense_client = None
    mongo_cursor = None  # Initialize cursor variable outside try block

    logger.info(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Counters for tracking progress
    total_docs_processed_mongo = 0
    total_docs_prepared = 0
    total_docs_skipped_preparation = 0
    total_docs_attempted_import = 0
    total_docs_imported_successfully = 0
    total_docs_failed_import = 0

    try:
        # --- 3. Connect to Services ---
        logger.info("Connecting to MongoDB...")
        print("Connecting to MongoDB...")
        mongo_client = MongoClient(
            MONGO_CONNECTION_STRING,
            serverSelectionTimeoutMS=10000  # Increased timeout slightly
        )
        # The ismaster command is cheap and does not require auth.
        mongo_client.admin.command('ismaster')
        mongo_db = mongo_client[DATABASE_NAME]
        mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
        logger.info(f"MongoDB connection successful. Accessing '{DATABASE_NAME}.{MONGO_COLLECTION_NAME}'.")
        print(f"MongoDB connection successful. Accessing '{DATABASE_NAME}.{MONGO_COLLECTION_NAME}'.")

        logger.info("Connecting to Typesense...")
        print("\nConnecting to Typesense...")
        typesense_client = typesense.Client({
            'nodes': [{
                'host': TYPESENSE_HOST,
                'port': TYPESENSE_PORT,
                'protocol': TYPESENSE_PROTOCOL
            }],
            'api_key': TYPESENSE_API_KEY,
            'connection_timeout_seconds': CONNECTION_TIMEOUT
        })

        # Verify connection and collection existence
        try:
            collection_info = typesense_client.collections[TYPESENSE_COLLECTION_NAME].retrieve()
            logger.info(f"Typesense connection successful. Target collection '{TYPESENSE_COLLECTION_NAME}' exists.")
            print(f"Typesense connection successful. Target collection '{TYPESENSE_COLLECTION_NAME}' exists.")
        except typesense.exceptions.ObjectNotFound:
            # This is a critical issue if you expect the collection to exist
            logger.error(f"CRITICAL Error: Typesense collection '{TYPESENSE_COLLECTION_NAME}' does not exist.")
            print(f"CRITICAL Error: Typesense collection '{TYPESENSE_COLLECTION_NAME}' does not exist.")
            print("Please create the collection in Typesense with the correct schema before running this script.")
            return {
                "success": False,
                "error": f"Typesense collection '{TYPESENSE_COLLECTION_NAME}' does not exist"
            }
        except typesense.exceptions.TypesenseClientError as e:
            # Catch other potential client errors during collection retrieval
            logger.error(f"Error accessing Typesense collection '{TYPESENSE_COLLECTION_NAME}': {e}")
            print(f"Error accessing Typesense collection '{TYPESENSE_COLLECTION_NAME}': {e}")
            return {
                "success": False,
                "error": f"Error accessing Typesense collection: {e}"
            }

        # --- 4. Fetch existing Typesense IDs ---
        logger.info("Fetching existing document IDs from Typesense...")
        print("\nFetching existing document IDs from Typesense...")
        existing_ids = set()
        try:
            # Get all document IDs from Typesense
            # Using pagination to handle large collections
            page_size = 250
            offset = 0
            while True:
                search_parameters = {
                    'q': '*',
                    'per_page': page_size,
                    'page': offset + 1,  # Typesense pages start at 1
                    'include_fields': 'id'
                }
                search_results = typesense_client.collections[TYPESENSE_COLLECTION_NAME].documents.search(search_parameters)
                
                # Get IDs from this page of results
                if 'hits' in search_results and search_results['hits']:
                    page_ids = [hit['document']['id'] for hit in search_results['hits']]
                    existing_ids.update(page_ids)
                    if len(page_ids) < page_size:
                        # Reached the end of results
                        break
                    offset += 1
                else:
                    # No more results
                    break
            
            logger.info(f"Found {len(existing_ids)} existing documents in Typesense.")
            print(f"Found {len(existing_ids)} existing documents in Typesense.")
        except Exception as e:
            logger.error(f"Error fetching existing Typesense IDs: {e}")
            print(f"Error fetching existing Typesense IDs: {e}")
            print("Will proceed with full import as fallback.")

        # --- 5. Fetch Data and Index in Batches ---
        logger.info("Starting data import from MongoDB to Typesense...")
        print(f"\nStarting data import from MongoDB to Typesense...")
        print(f"Batch Size: {BATCH_SIZE}, Action: '{IMPORT_ACTION}'")

        batch_number = 0
        batch = []
        start_import_time = time.time()

        # Modify MongoDB query to only get documents not in Typesense
        mongo_filter = {}
        if existing_ids:
            # Convert string IDs back to ObjectId for MongoDB query
            existing_object_ids = [ObjectId(id_str) for id_str in existing_ids if ObjectId.is_valid(id_str)]
            mongo_filter = {'_id': {'$nin': existing_object_ids}}
            logger.info(f"Filtering MongoDB query to only retrieve documents not in Typesense.")
            print(f"Filtering MongoDB query to only retrieve documents not in Typesense.")
        
        # Get total document count for progress estimation
        total_doc_count_mongo = mongo_collection.count_documents(mongo_filter)
        if total_doc_count_mongo == 0:
            logger.info("No new documents found in MongoDB collection. Exiting.")
            print("No new documents found in MongoDB collection. Exiting.")
            return {
                "success": True,
                "message": "No new documents to import",
                "total_docs_processed": 0,
                "total_docs_imported": 0
            }

        logger.info(f"Found {total_doc_count_mongo} new documents in MongoDB collection.")
        print(f"Found {total_doc_count_mongo} new documents in MongoDB collection.")
        # Estimate total batches, may change slightly if docs are skipped
        total_batches_estimated = math.ceil(total_doc_count_mongo / BATCH_SIZE)
        print(f"Estimated batches: {total_batches_estimated}")

        # Iterate through MongoDB documents using a cursor with filter
        mongo_cursor = mongo_collection.find(
            mongo_filter,  # Filter to only get new documents
            no_cursor_timeout=True  # Prevent cursor timeout for long operations
        )

        for mongo_doc in mongo_cursor:
            total_docs_processed_mongo += 1
            prepared_doc = prepare_document_for_typesense(mongo_doc)

            if prepared_doc:
                total_docs_prepared += 1
                batch.append(prepared_doc)
            else:
                total_docs_skipped_preparation += 1
                # Log skipping periodically if high volume
                if total_docs_skipped_preparation % 100 == 0:
                    logger.info(f"Skipped {total_docs_skipped_preparation} documents during preparation so far...")
                    print(f"Note: Skipped {total_docs_skipped_preparation} documents during preparation so far...")

            # Import when batch is full
            if len(batch) >= BATCH_SIZE:
                batch_number += 1
                logger.info(f"Processing Batch {batch_number}/{total_batches_estimated} (Size: {len(batch)})...")
                print(f"\nProcessing Batch {batch_number}/{total_batches_estimated} (Size: {len(batch)})...")
                batch_start_time = time.time()
                import_results = []  # Initialize results list for the batch

                try:
                    total_docs_attempted_import += len(batch)
                    import_results = typesense_client.collections[TYPESENSE_COLLECTION_NAME].documents.import_(
                        batch, {'action': IMPORT_ACTION}
                    )

                    # Process results carefully - import_results is a list of dicts
                    current_batch_success = 0
                    current_batch_failure = 0
                    for i, result_item in enumerate(import_results):
                        if isinstance(result_item, dict) and result_item.get('success'):
                            current_batch_success += 1
                        else:
                            current_batch_failure += 1
                            # Log details of the first few failures in a batch
                            if current_batch_failure <= 5:
                                logger.warning(f"Failure Detail (Batch {batch_number}, Item {i}): {result_item}")
                                print(f"  Failure Detail (Batch {batch_number}, Item {i}): {result_item}")

                    total_docs_imported_successfully += current_batch_success
                    total_docs_failed_import += current_batch_failure

                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    logger.info(f"Batch {batch_number} Import Results: {current_batch_success} successful, {current_batch_failure} failed. Duration: {batch_duration:.2f} seconds.")
                    print(f"Batch {batch_number} Import Results: {current_batch_success} successful, {current_batch_failure} failed.")
                    print(f"Batch {batch_number} duration: {batch_duration:.2f} seconds.")

                except typesense.exceptions.TypesenseClientError as e:
                    logger.error(f"API Error importing Batch {batch_number} to Typesense: {e}")
                    print(f"API Error importing Batch {batch_number} to Typesense: {e}")
                    # Attempt to log specific import errors if possible from the exception message
                    try:
                        error_details = json.loads(str(e))  # Assumes error message is JSON formatted
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                        print(f"Error details: {json.dumps(error_details, indent=2)}")
                    except (json.JSONDecodeError, TypeError):
                        logger.error(f"Raw error message: {e}")
                        print(f"Raw error message: {e}")  # Print raw error if not JSON

                    # Assume all documents in this specific batch failed due to the API error
                    total_docs_failed_import += len(batch)
                    logger.error(f"Assumed {len(batch)} failures for Batch {batch_number} due to API error.")
                    print(f"Assumed {len(batch)} failures for Batch {batch_number} due to API error.")

                except Exception as e:
                    # Catch unexpected errors during the import call itself
                    logger.error(f"Unexpected error during Typesense import for batch {batch_number}: {e}")
                    print(f"Unexpected error during Typesense import for batch {batch_number}: {e}")
                    total_docs_failed_import += len(batch)  # Assume batch failed
                    logger.error(f"Assumed {len(batch)} failures for Batch {batch_number} due to unexpected error.")
                    print(f"Assumed {len(batch)} failures for Batch {batch_number} due to unexpected error.")

                # Clear the batch for the next iteration
                batch = []

        # Process the final partial batch if any documents remain
        if batch:
            batch_number += 1
            logger.info(f"Processing Final Batch {batch_number} (Size: {len(batch)})...")
            print(f"\nProcessing Final Batch {batch_number} (Size: {len(batch)})...")
            batch_start_time = time.time()
            import_results = []
            try:
                total_docs_attempted_import += len(batch)
                import_results = typesense_client.collections[TYPESENSE_COLLECTION_NAME].documents.import_(
                    batch, {'action': IMPORT_ACTION}
                )

                current_batch_success = 0
                current_batch_failure = 0
                for i, result_item in enumerate(import_results):
                    if isinstance(result_item, dict) and result_item.get('success'):
                        current_batch_success += 1
                    else:
                        current_batch_failure += 1
                        if current_batch_failure <= 5:
                            logger.warning(f"Failure Detail (Final Batch, Item {i}): {result_item}")
                            print(f"  Failure Detail (Final Batch, Item {i}): {result_item}")

                total_docs_imported_successfully += current_batch_success
                total_docs_failed_import += current_batch_failure

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                logger.info(f"Final Batch Import Results: {current_batch_success} successful, {current_batch_failure} failed. Duration: {batch_duration:.2f} seconds.")
                print(f"Final Batch Import Results: {current_batch_success} successful, {current_batch_failure} failed.")
                print(f"Final Batch duration: {batch_duration:.2f} seconds.")

            except typesense.exceptions.TypesenseClientError as e:
                logger.error(f"API Error importing Final Batch to Typesense: {e}")
                print(f"API Error importing Final Batch to Typesense: {e}")
                try:
                    error_details = json.loads(str(e))
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    print(f"Error details: {json.dumps(error_details, indent=2)}")
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"Raw error message: {e}")
                    print(f"Raw error message: {e}")
                total_docs_failed_import += len(batch)
                logger.error(f"Assumed {len(batch)} failures for Final Batch due to API error.")
                print(f"Assumed {len(batch)} failures for Final Batch due to API error.")

            except Exception as e:
                logger.error(f"Unexpected error during final batch import: {e}")
                print(f"Unexpected error during final batch import: {e}")
                total_docs_failed_import += len(batch)
                logger.error(f"Assumed {len(batch)} failures for Final Batch due to unexpected error.")
                print(f"Assumed {len(batch)} failures for Final Batch due to unexpected error.")

        end_import_time = time.time()
        total_import_duration = end_import_time - start_import_time

        logger.info("\n--- Import Summary ---")
        logger.info(f"Total documents read from MongoDB: {total_docs_processed_mongo}")
        logger.info(f"Total documents successfully prepared for Typesense: {total_docs_prepared}")
        logger.info(f"Total documents skipped during preparation (missing ID/required fields): {total_docs_skipped_preparation}")
        logger.info(f"Total documents attempted to import to Typesense: {total_docs_attempted_import}")
        logger.info(f"Total documents successfully imported/upserted (reported by Typesense): {total_docs_imported_successfully}")
        logger.info(f"Total documents failed import (reported by Typesense or API errors): {total_docs_failed_import}")
        logger.info(f"Total import process duration: {total_import_duration:.2f} seconds")

        print("\n--- Import Summary ---")
        print(f"Total documents read from MongoDB: {total_docs_processed_mongo}")
        print(f"Total documents successfully prepared for Typesense: {total_docs_prepared}")
        print(f"Total documents skipped during preparation (missing ID/required fields): {total_docs_skipped_preparation}")
        print(f"Total documents attempted to import to Typesense: {total_docs_attempted_import}")
        print(f"Total documents successfully imported/upserted (reported by Typesense): {total_docs_imported_successfully}")
        print(f"Total documents failed import (reported by Typesense or API errors): {total_docs_failed_import}")
        print(f"Total import process duration: {total_import_duration:.2f} seconds")

        # Final sanity check
        if total_docs_prepared != total_docs_attempted_import:
            logger.warning(f"Warning: Mismatch between prepared docs ({total_docs_prepared}) and attempted imports ({total_docs_attempted_import}). Check logic.")
            print(f"Warning: Mismatch between prepared docs ({total_docs_prepared}) and attempted imports ({total_docs_attempted_import}). Check logic.")
        if total_docs_processed_mongo != total_docs_prepared + total_docs_skipped_preparation:
            logger.warning(f"Warning: Mismatch in MongoDB processing counts. Check logic.")
            print(f"Warning: Mismatch in MongoDB processing counts. Check logic.")

        return {
            "success": True,
            "total_docs_processed": total_docs_processed_mongo,
            "total_docs_prepared": total_docs_prepared,
            "total_docs_skipped": total_docs_skipped_preparation,
            "total_docs_imported": total_docs_imported_successfully,
            "total_docs_failed": total_docs_failed_import,
            "import_duration": total_import_duration
        }

    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        print(f"Error connecting to MongoDB: {e}")
        return {"success": False, "error": f"MongoDB connection error: {e}"}
    except pymongo.errors.OperationFailure as e:
        logger.error(f"MongoDB Operation Error (Authentication/Permissions issue?): {e}")
        print(f"MongoDB Operation Error (Authentication/Permissions issue?): {e}")
        return {"success": False, "error": f"MongoDB operation error: {e}"}
    except pymongo.errors.PyMongoError as e:
        # Errors during cursor iteration or other Mongo operations
        logger.error(f"A MongoDB error occurred during document processing: {e}")
        print(f"\nA MongoDB error occurred during document processing: {e}")
        return {"success": False, "error": f"MongoDB error during processing: {e}"}
    except typesense.exceptions.RequestMalformed as e:
        logger.error(f"Error connecting to Typesense (Malformed Request - check host/port/protocol?): {e}")
        print(f"Error connecting to Typesense (Malformed Request - check host/port/protocol?): {e}")
        return {"success": False, "error": f"Typesense connection error (malformed request): {e}"}
    except typesense.exceptions.AuthenticationError as e:
        logger.error(f"Error connecting to Typesense (Authentication Error - check API Key): {e}")
        print(f"Error connecting to Typesense (Authentication Error - check API Key): {e}")
        return {"success": False, "error": f"Typesense authentication error: {e}"}
    except typesense.exceptions.TypesenseClientError as e:
        # Catch-all for other typesense client errors during connection setup
        logger.error(f"General error connecting to Typesense: {e}")
        print(f"General error connecting to Typesense: {e}")
        return {"success": False, "error": f"Typesense client error: {e}"}
    except KeyboardInterrupt:
        logger.info("Import process interrupted by user (Ctrl+C).")
        print("\nImport process interrupted by user (Ctrl+C).")
        return {"success": False, "error": "Process interrupted by user"}
    except Exception as e:
        # Catch-all for any other unexpected errors during the main loop
        logger.error(f"An unexpected error occurred during the indexing process: {e}")
        print(f"\nAn unexpected error occurred during the indexing process: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback for unexpected errors
        return {"success": False, "error": f"Unexpected error: {e}"}
    finally:
        # --- 6. Close Connections ---
        if mongo_cursor:
            try:
                mongo_cursor.close()
                logger.info("MongoDB cursor closed.")
                print("\nMongoDB cursor closed.")
            except Exception as e:
                # Log error but don't prevent client closing
                logger.error(f"Error closing MongoDB cursor: {e}")
                print(f"Error closing MongoDB cursor: {e}")
        if mongo_client:
            try:
                mongo_client.close()
                logger.info("MongoDB connection closed.")
                print("MongoDB connection closed.")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")
                print(f"Error closing MongoDB connection: {e}")

        logger.info(f"Script finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Script finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    sync_mongodb_to_typesense() 