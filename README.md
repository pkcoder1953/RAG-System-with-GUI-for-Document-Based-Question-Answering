# RAG System with GUI for Document-Based Question Answering

## Overview
This project is a locally-runnable **Retrieval-Augmented Generation (RAG) System** designed to answer user queries based on content extracted from uploaded PDF documents. It processes text and images from PDF files and uses advanced NLP models to provide accurate, context-aware answers. The system features a GUI that displays responses, source pages, and relevant images interactively.

## Key Features
- **Locally Runnable**: No internet connection required after setup.
- **Free of Cost**: Fully open-source and cost-free to run locally.
- **Data Privacy**: Processes data locally without external servers, ensuring privacy.
- **Accurate Results**: Uses state-of-the-art models for text and context understanding.
- **Cache Implementation**:
  - JSON-based caching system to store query results.
  - Timestamps to expire entries after 24 hours.
  - MD5 hashing for efficient cache keys.
- **Multiple File Handling**: Processes multiple PDF files in one session.
- **Source Tracking**: Tracks and displays the origin of text and images.
- **Relevant Text and Image Generation**: Context-aware extraction of text and images for better answers.
- **GUI Implementation**: An interactive interface for viewing answers and images.
- **Interactive Query Loop**: Allows asking multiple questions in one session.
- **Error Handling and Debugging**: Robust error handling with user-friendly messages.
- **Optimized Image Handling**: Simplified processing to avoid memory issues.
- **Structured Output**: Provides answers with clear introduction, steps, and references.

## Example Query
**Query**: What is the step-by-step process of Direct Print Mode Setting?

**Output on GUI**:
- **Answer**: A detailed step-by-step guide for setting Direct Print Mode.
- **Relevant Images**: Displays images from the manual associated with the steps.
- **Source Pages**: Lists the pages from which the content was retrieved.



## How It Works
1. **PDF Processing**:
   - Extracts text and images from PDF files using `PyMuPDF`.
   - Chunks text for embedding generation.
   - Embeds extracted text using `SentenceTransformer`.
   - Stores images with surrounding context for relevance.

2. **Query Handling**:
   - Queries are embedded and compared with document embeddings to find relevant chunks.
   - Relevant text chunks are used to generate answers using `T5ForConditionalGeneration`.
   - Relevant images are identified by comparing the generated answer's embedding with image context embeddings.

3. **Cache**:
   - Query results are cached in a JSON file to avoid redundant processing.
   - Cache entries are automatically invalidated after 24 hours.

4. **GUI**:
   - Uses `tkinter` to display answers, source pages, and images in an interactive interface.

5. **Error Handling**:
   - Catches errors during processing and provides user-friendly messages.
   - Ensures smooth operation even with incomplete or poorly formatted PDFs.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - **NLP**: `transformers`, `sentence-transformers`
  - **PDF Processing**: `PyMuPDF`
  - **Machine Learning**: `scikit-learn`
  - **GUI**: `tkinter`, `Pillow`
  - **Utilities**: `os`, `hashlib`, `json`, `io`


## Usage
1. Launch the script and enter your queries in the terminal.
2. The GUI will pop up with the answer, source pages, and relevant images.
3. Interact with the GUI to view the results.
4. Repeat for multiple queries in the same session.

## Properties
- **Locally Runnable**: Fully self-contained; no external server required.
- **Data Privacy**: Ensures complete confidentiality of your documents.
- **Free of Cost**: Open-source and requires no subscription.
- **High Accuracy**: Leverages advanced NLP models for precise results.
- **Interactive**: Intuitive GUI for an enhanced user experience.

## Example Output
- **Query**: *"What is the process of loading paper?"*
- **Answer**: Detailed steps for loading paper with relevant images and source tracking.
- **GUI Screenshot**:
