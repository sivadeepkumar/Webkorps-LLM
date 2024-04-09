from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.core import Settings as settings
from llama_index.llms.openai import OpenAI
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
import os
import logging
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(filename='../logs/llama_openai_model.log', level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"]="true"

api_key = os.getenv("OPENAI_API_KEY")
documents = SimpleDirectoryReader("./sample_data/pdf_samples").load_data()
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.7,  # Increase the temperature for more creative responses
    max_tokens=2000,  # Increase the max_tokens for longer responses
    frequency_penalty=0.5,  # Adjust the frequency penalty as needed
    presence_penalty=0.5   # Adjust the presence penalty as needed
)

settings.chunk_size = 512
settings.llm = llm
settings.embed_model = embed_model

# Create Service Context using Settings

index = VectorStoreIndex.from_documents(documents, service_context=settings)
query_engine = index.as_query_engine()

app = Blueprint('llama_openai_model', __name__)

@app.route('/query', methods=['POST'])
def query():
    """
    Processes a query using the LLAMA OpenAI model and returns a response.

    This endpoint takes a POST request with a JSON payload containing the 'query' key representing the user's query. It processes the query using the LLAMA OpenAI model and returns the response in JSON format.

    Parameters:
        None (Request payload contains the 'query' key with the user's query string).

    Returns:
        JSON: A JSON response containing the processed response from the LLAMA OpenAI model.

    Raises:
        Exception: If any error occurs during the processing of the query, an error response with status code 500 is returned.
    """
    try:
        data = request.get_json()
        user_query = data.get('query')
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        
        response = query_engine.query(user_query)

        response_str = str(response.response)
        data = {"response":response}

        scores = [node.score for node in data['response'].source_nodes]
        print(scores)
        if scores[0] < 0.2:
            return jsonify({"response":"Please request the query related to documents"})

        return jsonify({"response": response_str})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred"}), 500
