import pymongo
import os
import json
from tqdm import tqdm
import sys
import logging
import tiktoken
from bson.objectid import ObjectId, InvalidId

# --- Configuration ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "")
DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "")
JSON_FOLDER_PATH = os.getenv("JSON_FOLDER_PATH", "parsed_json_results")
MONGO_ID_FIELD_NAME = os.getenv("MONGO_ID_FIELD_NAME", "mongo_object_id")
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", "8000"))  # Threshold for context window logic
# --- End Configuration ---

# Set up logging if not already configured
logger = logging.getLogger(__name__)

def process_single_document(data, filename, file_path, collection):
    """Process a single JSON document"""
    total_documents_inserted = 0
    files_with_errors = 0
    files_updated_with_ids = 0
    total_files_processed = 0
    
    document_updated = False
    
    # Skip documents without docId field
    if 'docId' not in data:
        logger.warning(f"Skipping document {filename}: No docId field found")
        files_with_errors += 1
        return False, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed
    
    # Create a query to find an identical document
    query = {}
    for key, value in data.items():
        if key not in ['_id', MONGO_ID_FIELD_NAME, 'context_list_ids']:
            query[key] = value
    
    # Check if an identical document exists (ignoring _id and mongo_object_id)
    try:
        existing_doc = collection.find_one(query)
        
        if existing_doc:
            # If document exists and is identical
            logger.info(f"Identical document found for {filename}. Using existing MongoDB ID.")
            # Add the existing MongoDB ID to the data
            data[MONGO_ID_FIELD_NAME] = str(existing_doc['_id'])
            document_updated = True
            return True, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed
        
    except Exception as e:
        logger.error(f"Error checking existing document for {filename}: {e}")
    
    # Insert document into MongoDB
    try:
        result = collection.insert_one(data)
        inserted_id = result.inserted_id
        total_documents_inserted += 1
        
        # Add inserted ID back to the data
        data[MONGO_ID_FIELD_NAME] = str(inserted_id)
        # Remove original _id field if present
        if '_id' in data:
            del data['_id']
        document_updated = True
        
    except pymongo.errors.DuplicateKeyError as dke:
        logger.error(f"Duplicate key error for {filename}: {dke}")
        files_with_errors += 1
    except Exception as e:
        logger.error(f"ERROR during insert for {filename}: {e}")
        files_with_errors += 1

    # Write the modified data back to the JSON file
    if document_updated:
        try:
            with open(file_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=4)
            files_updated_with_ids += 1
            total_files_processed += 1
            return True, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed
        except IOError as e:
            logger.error(f"ERROR writing updated JSON '{filename}': {e}")
            files_with_errors += 1
        except Exception as e:
            logger.error(f"ERROR during JSON write for '{filename}': {e}")
            files_with_errors += 1
    
    return False, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed

def calculate_tokens(text, encoding):
    """Calculate token count for a text using tiktoken"""
    if not isinstance(text, str):
        return 0
    token_ids = encoding.encode(text)
    return len(token_ids)

def process_multi_section_document(data_array, filename, file_path, collection):
    """Process a JSON file with multiple sections/chunks"""
    total_documents_inserted = 0
    files_with_errors = 0
    files_updated_with_ids = 0
    total_files_processed = 0
    
    modified = False
    valid_sections = []
    
    # Initialize tiktoken encoding
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.error(f"Error initializing tiktoken: {e}")
        encoding = None

    # First calculate total tokens and collect mongo_object_ids
    total_tokens = 0
    chunk_details = []  # List to store token counts and IDs for each chunk
    
    # Store original chunk numbers to preserve them
    original_chunk_numbers = {}
    for i, section in enumerate(data_array):
        if 'chunkNo' in section:
            original_chunk_numbers[i] = section['chunkNo']
    
    # If no chunk numbers exist, create them sequentially
    if not original_chunk_numbers:
        logger.info(f"No chunk numbers found in {filename}. Creating sequential chunk numbers.")
        for i, section in enumerate(data_array):
            original_chunk_numbers[i] = i + 1
            data_array[i]['chunkNo'] = i + 1
            modified = True
    
    # Skip sections without docId field
    for i, section in enumerate(data_array):
        # Skip sections without docId field
        if 'docId' not in section:
            logger.warning(f"Skipping section {i} in {filename}: No docId field found")
            continue
        
        valid_sections.append(i)
        
        # Calculate tokens for this section if embedding text exists
        chunk_tokens = 0
        if encoding and 'embedText' in section:
            chunk_tokens = calculate_tokens(section.get('embedText', ''), encoding)
        
        chunk_details.append({'index': i, 'tokens': chunk_tokens})
        total_tokens += chunk_tokens
    
    # Process each section/chunk as a separate document
    for i, section in enumerate(data_array):
        if i not in valid_sections:
            continue
            
        # Create a query to find an identical document
        query = {}
        for key, value in section.items():
            if key not in ['_id', MONGO_ID_FIELD_NAME, 'context_list_ids']:
                query[key] = value
        
        # Check if an identical document exists
        try:
            existing_doc = collection.find_one(query)
            
            if existing_doc:
                # If document exists and is identical
                logger.info(f"Identical document found for section {i} in {filename}. Using existing MongoDB ID.")
                # Add the existing MongoDB ID to the data
                data_array[i][MONGO_ID_FIELD_NAME] = str(existing_doc['_id'])
                modified = True
                continue
            
        except Exception as e:
            logger.error(f"Error checking existing document for section {i} in {filename}: {e}")
            
        try:
            result = collection.insert_one(section)
            inserted_id = result.inserted_id
            total_documents_inserted += 1
            
            # Add inserted ID back to the section
            data_array[i][MONGO_ID_FIELD_NAME] = str(inserted_id)
            # Remove original _id field if present
            if '_id' in data_array[i]:
                del data_array[i]['_id']
            modified = True
            
        except pymongo.errors.DuplicateKeyError as dke:
            logger.error(f"Duplicate key error for {filename} section {i}: {dke}")
            files_with_errors += 1
        except Exception as e:
            logger.error(f"ERROR during insert for {filename} section {i}: {e}")
            files_with_errors += 1
    
    # Check if any valid sections were processed
    if len(valid_sections) == 0:
        logger.warning(f"No valid sections (with docId) found in {filename}")
        files_with_errors += 1
        return False, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed
    
    # Add context_list_ids to each chunk based on token counts
    if encoding and modified:
        # Collect all valid mongo_object_ids
        all_mongo_ids = []
        for i in valid_sections:
            if MONGO_ID_FIELD_NAME in data_array[i]:
                all_mongo_ids.append(data_array[i][MONGO_ID_FIELD_NAME])
        
        # Apply context window logic based on total tokens
        if total_tokens < TOKEN_THRESHOLD:
            # For small documents, add all IDs to each chunk
            logger.info(f"File {filename} has {total_tokens} tokens (< {TOKEN_THRESHOLD}). Adding all IDs to each chunk.")
            for i in valid_sections:
                # Store original chunkNo
                original_chunk_no = data_array[i].get('chunkNo', i + 1)
                
                # Update context_list_ids 
                data_array[i]["context_list_ids"] = all_mongo_ids
                
                # Ensure chunkNo is preserved
                data_array[i]['chunkNo'] = original_chunk_no
                
                # Convert context IDs to ObjectIds in MongoDB directly
                if MONGO_ID_FIELD_NAME in data_array[i]:
                    current_id = data_array[i][MONGO_ID_FIELD_NAME]
                    try:
                        # Convert all context IDs to ObjectId objects for MongoDB
                        context_object_ids = [ObjectId(ctx_id) for ctx_id in all_mongo_ids]
                        
                        # Update the document in MongoDB with ObjectIds, preserving chunkNo
                        collection.update_one(
                            {"_id": ObjectId(current_id)},
                            {"$set": {
                                "context_list_ids": context_object_ids,
                                "chunkNo": original_chunk_no
                            }}
                        )
                        logger.info(f"Updated context_list_ids with ObjectIds for document {current_id}")
                    except Exception as e:
                        logger.error(f"Error updating context_list_ids with ObjectIds: {e}")
        else:
            # For large documents, create context windows
            logger.info(f"File {filename} has {total_tokens} tokens (>= {TOKEN_THRESHOLD}). Creating context windows.")
            for i in valid_sections:
                # Store original chunkNo
                original_chunk_no = data_array[i].get('chunkNo', i + 1)
                
                current_chunk_idx = i
                current_id = data_array[i].get(MONGO_ID_FIELD_NAME)
                
                if not current_id:
                    continue
                
                # Ensure chunkNo is preserved
                data_array[i]['chunkNo'] = original_chunk_no
                
                context_window_ids = [current_id]
                current_window_tokens = chunk_details[valid_sections.index(i)]['tokens']
                
                # Expand backward
                j = valid_sections.index(i) - 1
                while j >= 0 and current_window_tokens < TOKEN_THRESHOLD:
                    prev_idx = valid_sections[j]
                    prev_tokens = chunk_details[j]['tokens']
                    prev_id = data_array[prev_idx].get(MONGO_ID_FIELD_NAME)
                    
                    if prev_id and (current_window_tokens + prev_tokens) < TOKEN_THRESHOLD:
                        current_window_tokens += prev_tokens
                        context_window_ids.insert(0, prev_id)
                    else:
                        break
                    j -= 1
                
                # Expand forward
                k = valid_sections.index(i) + 1
                while k < len(valid_sections) and current_window_tokens < TOKEN_THRESHOLD:
                    next_idx = valid_sections[k]
                    next_tokens = chunk_details[k]['tokens']
                    next_id = data_array[next_idx].get(MONGO_ID_FIELD_NAME)
                    
                    if next_id and (current_window_tokens + next_tokens) < TOKEN_THRESHOLD:
                        current_window_tokens += next_tokens
                        context_window_ids.append(next_id)
                    else:
                        break
                    k += 1
                
                data_array[i]["context_list_ids"] = context_window_ids
                
                # Convert context IDs to ObjectIds in MongoDB directly
                try:
                    # Convert all context IDs to ObjectId objects for MongoDB
                    context_object_ids = [ObjectId(ctx_id) for ctx_id in context_window_ids]
                    
                    # Update the document in MongoDB with ObjectIds, preserving chunkNo
                    collection.update_one(
                        {"_id": ObjectId(current_id)},
                        {"$set": {
                            "context_list_ids": context_object_ids,
                            "chunkNo": original_chunk_no
                        }}
                    )
                    logger.info(f"Updated context_list_ids with ObjectIds for document {current_id}")
                except Exception as e:
                    logger.error(f"Error updating context_list_ids with ObjectIds: {e}")
    
    # Write the modified data back to the JSON file if any section was updated
    if modified:
        try:
            # Restore original chunk numbers before writing back to file
            for i, chunk_num in original_chunk_numbers.items():
                if i < len(data_array):
                    data_array[i]['chunkNo'] = chunk_num
            
            with open(file_path, 'w', encoding='utf-8') as f_out:
                json.dump(data_array, f_out, indent=4)
            files_updated_with_ids += 1
            total_files_processed += 1
            return True, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed
        except IOError as e:
            logger.error(f"ERROR writing updated JSON '{filename}': {e}")
            files_with_errors += 1
        except Exception as e:
            logger.error(f"ERROR during JSON write for '{filename}': {e}")
            files_with_errors += 1
    
    return False, total_documents_inserted, files_with_errors, files_updated_with_ids, total_files_processed

def upload_documents(json_folder_path=JSON_FOLDER_PATH, mongo_connection=MONGO_CONNECTION_STRING, 
                     db_name=DATABASE_NAME, collection_name=COLLECTION_NAME, 
                     id_field_name=MONGO_ID_FIELD_NAME):
    """
    Upload JSON documents to MongoDB and update JSON files with MongoDB IDs.
    
    Args:
        json_folder_path: Path to the folder containing JSON files
        mongo_connection: MongoDB connection string
        db_name: MongoDB database name
        collection_name: MongoDB collection name
        id_field_name: Name of the field to store MongoDB ID in JSON files
        
    Returns:
        dict: Results summary with success and document counts
    """
    logger.info("==== Starting MongoDB Upload Process ====")
    
    total_files_processed = 0
    total_documents_inserted = 0
    files_with_errors = 0
    files_updated_with_ids = 0
    
    # Check if tiktoken is available
    try:
        import tiktoken
        logger.info("Tiktoken library available for token counting")
    except ImportError:
        logger.warning("Tiktoken library not found. Token counting for context windows will be disabled.")
    
    # --- MongoDB Connection ---
    client = None
    try:
        logger.info(f"Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_connection, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logger.info("MongoDB connection successful.")
        db = client[db_name]
        collection = db[collection_name]
        logger.info(f"Using database: '{db_name}', collection: '{collection_name}'")
        try:
           count = collection.count_documents({})
           logger.info(f"Collection '{collection_name}' currently has {count} documents.")
        except Exception as e:
           logger.error(f"Could not get document count (collection might be new): {e}")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        logger.error(f"MongoDB connection failed: Server selection timeout - {err}")
        return {"success": False, "error": f"MongoDB connection failed: {err}"}
    except pymongo.errors.ConnectionFailure as err:
        logger.error(f"MongoDB connection failed: {err}")
        return {"success": False, "error": f"MongoDB connection failed: {err}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
        return {"success": False, "error": f"MongoDB connection error: {e}"}

    # --- Process JSON Files ---
    logger.info(f"Processing and updating JSON files from: {json_folder_path}")
    logger.info(f"WARNING: This script will modify JSON files in place in '{json_folder_path}' to add the '{id_field_name}' field.")
    
    try:
        # Find all directories in the JSON_FOLDER_PATH
        json_dirs = [d for d in os.listdir(json_folder_path) if os.path.isdir(os.path.join(json_folder_path, d))]
        
        if not json_dirs:
            logger.warning("No subdirectories found in the specified directory.")
        else:
            logger.info(f"Found {len(json_dirs)} subdirectories.")
            
            all_json_files = []
            
            # Collect all JSON files from subdirectories
            for directory in json_dirs:
                dir_path = os.path.join(json_folder_path, directory)
                json_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.json')]
                all_json_files.extend(json_files)
            
            logger.info(f"Found {len(all_json_files)} JSON files in total.")
            
            for file_path in tqdm(all_json_files, desc="Uploading & Updating JSON"):
                filename = os.path.basename(file_path)
                
                try:
                    # Read JSON file data
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if it's a single document or multi-section format
                    if isinstance(data, list):
                        # Multi-section document (like ISM Manual)
                        success, docs_inserted, files_errors, files_updated, files_processed = process_multi_section_document(data, filename, file_path, collection)
                        total_documents_inserted += docs_inserted
                        files_with_errors += files_errors
                        files_updated_with_ids += files_updated
                        total_files_processed += files_processed
                    elif isinstance(data, dict):
                        # Single document (like Fleet Alert)
                        success, docs_inserted, files_errors, files_updated, files_processed = process_single_document(data, filename, file_path, collection)
                        total_documents_inserted += docs_inserted
                        files_with_errors += files_errors
                        files_updated_with_ids += files_updated
                        total_files_processed += files_processed
                    else:
                        logger.warning(f"Expected dict or list in '{filename}', found {type(data)}. Skipping.")
                        files_with_errors += 1
                        
                except FileNotFoundError:
                    logger.error(f"File not found: {file_path}")
                    files_with_errors += 1
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON format in file: {filename}")
                    files_with_errors += 1
                except Exception as e:
                    logger.error(f"An unexpected error occurred processing file {filename}: {e}")
                    files_with_errors += 1

    except FileNotFoundError:
        logger.error(f"Error: The directory '{json_folder_path}' does not exist.")
        return {"success": False, "error": f"Directory not found: {json_folder_path}"}
    except Exception as e:
        logger.error(f"An error occurred while listing files: {e}")
        return {"success": False, "error": f"Error processing files: {e}"}
    finally:
        # Close MongoDB connection
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

    # --- Summary ---
    logger.info("\n--- Upload & Update Summary ---")
    logger.info(f"JSON files processed (read attempt): {total_files_processed + files_with_errors}")
    logger.info(f"JSON files successfully updated with MongoDB IDs: {files_updated_with_ids}")
    logger.info(f"Total documents inserted into MongoDB: {total_documents_inserted}")
    logger.info(f"Files skipped or encountering errors (read/insert/write): {files_with_errors}")
    logger.info(f"Target Database: '{db_name}'")
    logger.info(f"Target Collection: '{collection_name}'")
    logger.info(f"MongoDB ID field added to JSONs: '{id_field_name}'")
    
    return {
        "success": total_documents_inserted > 0 or files_updated_with_ids > 0,
        "total_files_processed": total_files_processed + files_with_errors,
        "files_updated_with_ids": files_updated_with_ids,
        "documents_inserted": total_documents_inserted,
        "files_with_errors": files_with_errors
    }

def update_context_ids_in_mongodb(json_folder_path, mongo_connection=MONGO_CONNECTION_STRING, 
                                db_name=DATABASE_NAME, collection_name=COLLECTION_NAME):
    """
    Updates MongoDB documents with context_list_ids from processed JSON files.
    This function is typically run after the initial upload to ensure context_list_ids
    are properly stored in MongoDB as ObjectId objects, not strings.
    """
    logger.info("==== Starting MongoDB Context ID Update Process ====")
    
    # --- MongoDB Connection ---
    client = None
    try:
        logger.info(f"Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_connection, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logger.info("MongoDB connection successful.")
        db = client[db_name]
        collection = db[collection_name]
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return {"success": False, "error": f"MongoDB connection failed: {e}"}
    
    files_processed = 0
    chunks_processed = 0
    updates_attempted = 0
    updates_successful = 0
    chunks_skipped = 0
    errors = 0
    
    # First check some documents to see if they already have ObjectId type context_list_ids
    try:
        # Try to find a document with context_list_ids
        sample_doc = collection.find_one({"context_list_ids": {"$exists": True}})
        if sample_doc and "context_list_ids" in sample_doc:
            # Check the type of the first ID in the list
            if sample_doc["context_list_ids"] and isinstance(sample_doc["context_list_ids"][0], ObjectId):
                logger.info("Found document with ObjectId type context_list_ids. No conversion needed.")
                return {
                    "success": True,
                    "message": "Documents already have ObjectId type context_list_ids"
                }
            else:
                logger.info("Found document with string type context_list_ids. Converting to ObjectIds.")
    except Exception as e:
        logger.error(f"Error checking document types: {e}")
    
    try:
        # Process all JSON files
        for root, dirs, files in os.walk(json_folder_path):
            for filename in files:
                if not filename.lower().endswith('.json'):
                    continue
                    
                file_path = os.path.join(root, filename)
                files_processed += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if not isinstance(data, list):
                        # Only process multi-section documents
                        continue
                        
                    for chunk in data:
                        chunks_processed += 1
                        
                        if not isinstance(chunk, dict):
                            chunks_skipped += 1
                            continue
                            
                        mongo_id = chunk.get(MONGO_ID_FIELD_NAME)
                        context_list_ids = chunk.get("context_list_ids")
                        
                        if not mongo_id or not context_list_ids:
                            chunks_skipped += 1
                            continue
                            
                        # Convert string IDs to ObjectId for MongoDB
                        context_object_ids = []
                        valid_ids = True
                        for ctx_id in context_list_ids:
                            try:
                                # Convert each string ID to ObjectId
                                obj_id = ObjectId(ctx_id)
                                context_object_ids.append(obj_id)
                            except InvalidId:
                                logger.warning(f"Invalid ObjectId format in context_list_ids: {ctx_id} - skipping update")
                                valid_ids = False
                                break
                            except Exception as e:
                                logger.warning(f"Error converting ID {ctx_id} to ObjectId: {e} - skipping update")
                                valid_ids = False
                                break
                                
                        if not valid_ids:
                            chunks_skipped += 1
                            continue
                                
                        updates_attempted += 1
                        
                        try:
                            # Update MongoDB document with context IDs (converted to ObjectIds)
                            # First convert the document ID to ObjectId
                            doc_id = ObjectId(mongo_id)
                            
                            # IMPORTANT: Force MongoDB to store ObjectIds explicitly
                            # First verify this document exists
                            doc_exists = collection.find_one({"_id": doc_id})
                            if not doc_exists:
                                logger.warning(f"No document found with _id {mongo_id}")
                                chunks_skipped += 1
                                continue
                                
                            # Perform the update with the list of ObjectIds using direct BSON construction
                            result = collection.update_one(
                                {"_id": doc_id},
                                {"$set": {"context_list_ids": context_object_ids}}
                            )
                            
                            if result.matched_count > 0:
                                if result.modified_count > 0:
                                    updates_successful += 1
                                    logger.info(f"Successfully updated context_list_ids for document {mongo_id}")
                                else:
                                    logger.info(f"Document {mongo_id} matched but not modified (possibly already has identical context_list_ids)")
                            else:
                                logger.warning(f"No document found with _id {mongo_id}")
                                chunks_skipped += 1
                                
                        except Exception as e:
                            logger.error(f"Error updating MongoDB document {mongo_id}: {e}")
                            errors += 1
                            
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    errors += 1
                    
                if files_processed % 50 == 0:
                    logger.info(f"Processed {files_processed} files...")
                    
        # Verify the updates
        sample_count = min(5, updates_successful)
        if sample_count > 0:
            logger.info(f"Verifying ObjectId conversion for {sample_count} random documents...")
            try:
                sample_docs = list(collection.find({"context_list_ids": {"$exists": True}}).limit(sample_count))
                for doc in sample_docs:
                    if "context_list_ids" in doc and doc["context_list_ids"]:
                        id_type = type(doc["context_list_ids"][0])
                        logger.info(f"Document {doc['_id']} has context_list_ids of type: {id_type}")
                        if not isinstance(doc["context_list_ids"][0], ObjectId):
                            logger.warning(f"Document {doc['_id']} still has non-ObjectId context_list_ids")
            except Exception as e:
                logger.error(f"Error verifying ObjectId conversion: {e}")
    
    except Exception as e:
        logger.error(f"An error occurred during context ID update: {e}")
        return {"success": False, "error": f"Error during context ID update: {e}"}
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")
    
    # --- Summary ---
    logger.info("\n--- Context ID Update Summary ---")
    logger.info(f"JSON files processed: {files_processed}")
    logger.info(f"Chunks processed: {chunks_processed}")
    logger.info(f"Updates attempted: {updates_attempted}")
    logger.info(f"Updates successful: {updates_successful}")
    logger.info(f"Chunks skipped: {chunks_skipped}")
    logger.info(f"Errors encountered: {errors}")
    
    return {
        "success": updates_successful > 0,
        "files_processed": files_processed,
        "updates_successful": updates_successful,
        "errors": errors
    }

if __name__ == "__main__":
    # Configure basic logging when running as a script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mongodb_upload.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run the upload function
    result = upload_documents()
    
    # Update context IDs in MongoDB
    if result["success"]:
        logger.info("Running context ID update in MongoDB...")
        update_result = update_context_ids_in_mongodb(JSON_FOLDER_PATH)
    
    # Print summary to console
    print("\n--- Upload & Update Summary ---")
    print(f"JSON files processed (read attempt): {result.get('total_files_processed', 0)}")
    print(f"JSON files successfully updated with MongoDB IDs: {result.get('files_updated_with_ids', 0)}")
    print(f"Total documents inserted into MongoDB: {result.get('documents_inserted', 0)}")
    print(f"Files skipped or encountering errors (read/insert/write): {result.get('files_with_errors', 0)}")
    print(f"Target Database: '{DATABASE_NAME}'")
    print(f"Target Collection: '{COLLECTION_NAME}'")
    print(f"MongoDB ID field added to JSONs: '{MONGO_ID_FIELD_NAME}'")
    print("---------------------------------") 