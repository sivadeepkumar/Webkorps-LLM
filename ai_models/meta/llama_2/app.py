from flask import request, jsonify, Blueprint
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from llama_index.core.settings import Settings  # Import Settings class
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb

logging.basicConfig(filename='../logs/llama2_model.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# load documents
documents = SimpleDirectoryReader("./sample_data/pdf_samples/").load_data()

# Initialize HuggingFace Embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5") #-- We can also use this model
pdb.set_trace()
# Initialize LLama2 model (load it only once)
llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01, 'context_length': 1800},
                   )

# Set LLama model and embeddings in the Settings class
Settings.llm = llm
Settings.embed_model = embed_model

# Initialize VectorStoreIndex with LLama2 and HuggingFace Embeddings
index = VectorStoreIndex.from_documents(documents,service_context=Settings)

# Initialize Query Engine
query_engine = index.as_query_engine()

app = Blueprint('llama_openai_model', __name__)

@app.route('/query', methods=['POST'])
def generate_response():
    """
    Generates a response to a user query using the LLama2 model.

    This endpoint takes a POST request with a JSON payload containing the 'query' key representing the user's query. It processes the query using the LLama2 model and returns the response in JSON format.

    Parameters:
        None (Request payload contains the 'query' key with the user's query string).

    Returns:
        JSON: A JSON response containing the generated response from the LLama2 model.

    Raises:
        Exception: If any error occurs during the processing of the query, an error response with status code 500 is returned.
    """

    try:
        # Get the query from the request body
        data = request.get_json()
        user_query = data.get('query')
        print(user_query)
        # Check if the query is provided
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        response = query_engine.query(user_query)
        response_str = str(response.response)

        #   How to get the response here in string format
        return jsonify({"response": response_str})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred"}), 500
