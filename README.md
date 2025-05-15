# Document Automation System

This system automates the process of scraping, converting, parsing, and uploading different types of documents to cloud storage and databases.

## Features

- Scrape documents from a website
- Download files to organized folders
- Convert non-PDF files to PDF format
- Parse different document types to JSON:
  - ISM Manual documents using LlamaIndex and LlamaParse
  - Office Manual documents using LlamaIndex and LlamaParse
  - Various other document types using OpenAI.
    - Office Forms
    - RA Database
    - Forms and Checklists
    - Standard eForm
    - Fleet Alert
- Add metadata to JSON files from CSV sources
- Create document headers and embedding text for better search and retrieval
- Add sequential chunk numbers to processed documents
- Upload documents to Amazon S3 for storage and accessibility
- Store document data in MongoDB with references to S3 files
- Skip already processed documents to avoid redundant processing

## Processing Pipeline

The system implements a multi-stage document processing pipeline:

1. **Document Scraping** (`document_scrap.py`)
   - Scrapes documents from configured websites
   - Organizes files by document type

2. **Document Conversion** (`document_to_pdf.py`)
   - Converts non-PDF files to PDF format
   - Uses Adobe PDF Services API

3. **Document Parsing** (Multiple modules)
   - Type-specific parsers for different document formats
   - Extracts structured data to JSON
   - Skips documents that already have corresponding JSON files

4. **LlamaParse Processing** (`llamaparse_processing.py`)
   - Uses LlamaParse for efficient PDF parsing
   - Integrates with OpenAI or Anthropic for content extraction
   - Generates comprehensive and short summaries

5. **Post-Processing** (`post_processing.py`)
   - Flattens page structures to sections
   - Normalizes document metadata
   - Handles continued sections across pages

6. **Metadata Updates** (`update_json_with_metadata.py`)
   - Enriches parsed JSON with additional metadata from CSV files
   - Creates document headers and embedding text fields
   - Standardizes field names across document types

7. **Chunk Numbering** (Part of `post_processing.py`)
   - Adds sequential chunk numbers to each item in processed JSON files
   - Facilitates tracking and reference of document chunks

8. **S3 Upload** (`document_s3_uploader.py`)
   - Uploads PDF and original document files to Amazon S3
   - Creates organized folder structure in S3 based on document types
   - Updates JSON files with S3 URLs for document access

9. **MongoDB Upload** (`upload_to_mongodb.py`)
   - Stores document JSON data in MongoDB collection
   - Handles both single documents and multi-section documents
   - Adds MongoDB object IDs back to JSON files for reference

10. **Typesense Sync** (`mongodb_to_typesense.py`)
    - Indexes MongoDB documents in Typesense
    - Enables fast and efficient search capabilities

## Document Parser Modules

The system includes the following document parsers:

1. **ISM Manual Parser (`document_parser_ism.py`)**
   - Specifically optimized for parsing ISM (International Safety Management) manuals
   - Uses LlamaIndex and LlamaParse for document parsing
   - Extracts chapters, sections, and document metadata

2. **Office Manual Parser (`document_parser_manual.py`)**
   - Specifically optimized for parsing Office Manuals
   - Uses LlamaIndex and LlamaParse for document parsing
   - Extracts sections, subsections, and document metadata

3. **General Document Parser (`document_parser_general.py`)**
   - Handles various other document types
   - Uses OpenAI and LlamaParse for document parsing

## Command Line Usage

The main script accepts various command line arguments to control the operation:

### Basic Usage

```
python main.py
```

By default, without any arguments, the script will:
1. Process all document types (ISM Manual, Office Manual, Fleet Alert, Forms and Checklists, etc.)
2. Execute the complete pipeline (scraping, conversion, parsing, post-processing)
3. Upload to S3, MongoDB, and sync to Typesense
4. Skip documents that already have corresponding JSON files

### Document Parsing Options

To parse specific document types:

```
# Parse only ISM Manual documents
python main.py --parse-ism-only

# Parse only Office Manual documents
python main.py --parse-doc-type="Office Manual"

# Parse any other specific document type handled by the general parser
python main.py --parse-doc-type=<document_type>

# Parse all document types (includes S3 and MongoDB upload)
python main.py --only-parse --parse-doc-type=all

# Only parse documents (skip scraping and conversion)
python main.py --only-parse --parse-doc-type=<document_type>
```

### Metadata and Post-Processing Options

```
# Only add chunk numbers to existing JSON files
python main.py --only-add-chunks

# Run the standalone metadata update script
python update_json_with_metadata.py
```

### Cloud Storage and Database Options

```
# Only upload documents to S3 (standalone)
python document_s3_uploader.py

# Only upload JSON files to MongoDB (standalone)
python upload_to_mongodb.py
```

### LlamaParse Options

```
# Use LlamaParse for document parsing
python main.py --use-llamaparse

# Specify LLM vendor for LlamaParse
python main.py --use-llamaparse --vendor-model="anthropic-sonnet-3.7"

# Test LlamaParse on a specific document
python test_llama_parse.py <path_to_pdf>
```

### Other Options

```
# Only scrape documents
python main.py --only-scrape

# Only convert documents to PDF
python main.py --only-convert

# Specify output directories
python main.py --output-dir=<directory> --json-output-dir=<directory>

# Control concurrency
python main.py --max-workers=<number>
```

### Authentication and API Keys

```
# Provide credentials for website scraping
python main.py --username=<username> --password=<password>

# Provide Adobe API credentials for PDF conversion
python main.py --adobe-id=<client_id> --adobe-secret=<client_secret>

# Provide OpenAI API key for document parsing
python main.py --openai-api-key=<api_key>

# Provide LlamaParse API key
python main.py --llama-api-key=<api_key>
```

## Environment Variables

For security, you can set credentials in a `.env` file:

```
SCRAPER_USERNAME=your_username
SCRAPER_PASSWORD=your_password
SCRAPER_DOMAIN=docMap

DOWNLOADER_USERNAME=your_download_username
DOWNLOADER_PASSWORD=your_download_password

ADOBE_CLIENT_ID=your_adobe_client_id
ADOBE_CLIENT_SECRET=your_adobe_client_secret

OPENAI_API_KEY=your_openai_api_key
LLAMA_API_KEY=your_llama_api_key
VENDOR_API_KEY=your_anthropic_api_key
VENDOR_MODEL=anthropic-sonnet-3.7

# S3 Configuration (optional if using AWS credentials file)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1

# MongoDB Configuration
MONGO_CONNECTION_STRING=mongodb://username:password@host:port/database
MONGO_DATABASE=database_name
MONGO_COLLECTION=collection_name

OUTPUT_DIRECTORY=downloaded_data
JSON_OUTPUT_DIR=parsed_json_results
MAX_WORKERS=3
```

## Directory Structure

The system creates the following directory structure for downloaded and processed files:

```
output_dir/
├── ISM Manual/
│   └── [PDF files]
├── Office Manual/
│   └── [PDF files]
└── [Other document type folders]/
    └── [PDF files]

json_output_dir/
├── ISM Manual/
│   └── [JSON files]
├── Office Manual/
│   └── [JSON files]
└── [Other document type folders]/
    └── [JSON files]
```

## Cloud Storage and Database Structure

### Amazon S3

Documents are organized in S3 with the following structure:
```
syia_documents/
├── ISM_Manual/
│   ├── ISM Manual Pdf/
│   │   └── [PDF files]
│   └── ISM Manual original Format/
│       └── [Original files]
├── Office_Manual/
│   ├── Office Manual Pdf/
│   │   └── [PDF files]
│   └── Office Manual original Format/
│       └── [Original files]
└── [Other document type folders]/
    ├── [Document Type] Pdf/
    │   └── [PDF files]
    └── [Document Type] original Format/
        └── [Original files]
```

### MongoDB

Document data is stored in MongoDB with the following structure:
- ISM Manual documents: Each section in a document becomes a separate MongoDB document
- Fleet Alert and other documents: Each document is stored as a single MongoDB document
- Each document or section in MongoDB gets a unique ID that is referenced in the JSON files 

# MongoDB and Typesense Sync

This project contains scripts to upload documents to MongoDB and synchronize them with Typesense.

## Environment Variables

Before running the scripts, you need to set the following environment variables:

### MongoDB Configuration
- `MONGO_CONNECTION_STRING`: MongoDB connection string (e.g. "mongodb://user:password@host/?authSource=database")
- `MONGO_DATABASE_NAME`: MongoDB database name
- `MONGO_COLLECTION_NAME`: MongoDB collection name

### Typesense Configuration
- `TYPESENSE_API_KEY`: Typesense API key
- `TYPESENSE_HOST`: Typesense host (e.g. "xxx.a1.typesense.net")
- `TYPESENSE_PORT`: Typesense port (default: 443)
- `TYPESENSE_PROTOCOL`: Typesense protocol (default: "https")
- `TYPESENSE_COLLECTION_NAME`: Typesense collection name

### Additional Configuration
- `JSON_FOLDER_PATH`: Path to folder containing JSON files to upload (default: "parsed_json_results")
- `MONGO_ID_FIELD_NAME`: Field name to store MongoDB ID in JSON files (default: "mongo_object_id")
- `TOKEN_THRESHOLD`: Threshold for context window logic (default: 8000)
- `BATCH_SIZE`: Number of documents to upload in a batch (default: 100)
- `IMPORT_ACTION`: Import action for Typesense (default: "upsert")
- `CONNECTION_TIMEOUT`: Connection timeout in seconds (default: 60)

## Setting Environment Variables

### Unix/Linux/macOS
```bash
export MONGO_CONNECTION_STRING="mongodb://user:password@host/?authSource=database"
export MONGO_DATABASE_NAME="your-database"
export MONGO_COLLECTION_NAME="your-collection"
export TYPESENSE_API_KEY="your-api-key"
export TYPESENSE_HOST="your-typesense-host"
# Set other variables as needed
```

### Windows
```cmd
set MONGO_CONNECTION_STRING=mongodb://user:password@host/?authSource=database
set MONGO_DATABASE_NAME=your-database
set MONGO_COLLECTION_NAME=your-collection
set TYPESENSE_API_KEY=your-api-key
set TYPESENSE_HOST=your-typesense-host
REM Set other variables as needed
```

## Running the Scripts

1. Upload documents to MongoDB:
```
python upload_to_mongodb.py
```

2. Sync MongoDB documents to Typesense:
```
python mongodb_to_typesense.py
``` # UTOMATED-DOCUMENT-PARSING-RAG
