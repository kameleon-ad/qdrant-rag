# Nice Classification with RAG

## Requirements
- **Python**: Runtime Platform
- **OpenAI**: Embedding of the text & Inference of the class
- **Qdrant**: Vector Database

## Prepare the environment
1. Set up virtual environment for runtime
   - Linux or Macos
      ```python
      python -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      ```
   - Windows
      ```python
      python -m venv .venv
      .venv/Scripts/activate
      pip install -r requirements.txt
      ```
2. Configure the environment variables
   - Create a new `.env` file.
   - Edit `.env` file
      ```env
      OPENAI_API_KEY="OPENAI_API_KEY"
      OPENAI_ORGANIZATION_KEY="OPENAI_ORGANIZATION_KEY"

      QDRANT_API_KEY="QDRANT_API_KEY"
      QDRANT_CLUSTER="QDRANT_CLUSTER"
       ```

## How to prepare data
1. Parse XML files to prepare the data
   ```bash
   python xml_parse.py
   ```
2. Prepare the Qdrant and prepare the vector database
   ```bash
   python data_retrieve.py data/output.json
   ```

## Run the server
1. Run the flask server
   ```bash
   python -m flask run
   ```
2. Results
![Screenshot from 2024-06-14 05-35-06](imgs/Screenshot%20from%202024-06-14%2005-35-06.png "Screenshot 1")
![Screenshot from 2024-06-14 05-35-53](imgs/Screenshot%20from%202024-06-14%2005-35-53.png "Screenshot 2")
