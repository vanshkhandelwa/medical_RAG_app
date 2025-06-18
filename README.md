# Medical RAG QA System using Meditron 7B LLM

This project implements a Medical Question-Answering system using Retrieval-Augmented Generation (RAG) with the Meditron 7B LLM model. The system uses Chroma as the vector database and PubMedBERT for embeddings.

## Features

- Medical document ingestion and processing
- Vector-based semantic search using Chroma
- Question answering powered by Meditron 7B LLM
- Web interface for easy interaction
- PubMedBERT embeddings for medical text

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd Medical-RAG-using-Meditron-7B-LLM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Meditron model:
```bash
python download_model.py
```

5. Ingest your medical documents:
```bash
python ingest.py
```

6. Run the application:
```bash
python rag.py
```

## Project Structure

- `rag.py`: Main application file with web interface
- `ingest.py`: Document ingestion and processing
- `retriever.py`: Vector search implementation
- `download_model.py`: Script to download Meditron model
- `templates/`: HTML templates for web interface
- `static/`: Static files (CSS, JS)
- `chroma_db/`: Vector database storage
- `models/`: Downloaded model files

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
