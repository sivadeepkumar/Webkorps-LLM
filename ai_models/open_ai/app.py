from flask import Flask, request, jsonify, g
from .openai_service import *
from dotenv import load_dotenv
from flask_cors import CORS 
import os
import logging
from flask import Blueprint
load_dotenv()

app = Blueprint('open_ai_model', __name__)

# Configure logging
logging.basicConfig(filename='../../logs/open_ai_model.log', level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route('/', methods=['GET'])
def health_check():
    return 'OK', 200


@app.route('/cryoport', methods=['POST'])
def cryoport():
    """_Processes a query related to Cryoport information.
        This endpoint is designed to handle POST requests and processes queries regarding Cryoport. Upon receiving a valid JSON payload containing a 'query' field, the endpoint extracts the query and processes it to retrieve relevant information about Cryoport and its uses.
        Returns:
        JSON: Processed data related to the query, typically information about Cryoport and its uses.
        The processed data is then returned in JSON format, providing insights and details based on the query submitted to the endpoint.
    """

    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('cryoport_text.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /cryoport endpoint."+str(e))
        return jsonify({'Error': 'Internal Server Error'}), 500

@app.route('/realEstateQuery', methods=['POST'])
def realEstateQuery():
    """
    Processes queries related to real estate information.
    This function handles POST requests containing a JSON payload with a 'query' field. It processes the query using the provided 'estate.txt' data file, which includes information about the Purva Park Hill real estate project. 
    The processed data is returned in JSON format.
    Returns:
        JSON: Processed data related to the query about real estate.
    """

    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('estate.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /realEstateQuery endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/query', methods=['POST'])
def assetpanda():
    """
    Processes queries related to AssetPanda information.
    This function handles POST requests containing a JSON payload with a 'query' field. It processes the query using the provided 'assetpanda.txt' data file, which includes information about adding a record, viewing a record, and tracking a record in AssetPanda. 
    The processed data is returned in JSON format.
    Returns:
        JSON: Processed data related to the query about AssetPanda.
    """

    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('assetpanda.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /query (ASSETPANDA) endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/webkorps_query', methods=['POST'])
def webkorps_query():
    """
    Processes queries related to Webkorps information.
    This function handles POST requests containing a JSON payload with a 'query' field. It processes the query using the provided 'webkorps_data.txt' data file, which includes information about Webkorps Services India Pvt Ltd, specifically focusing on the Leave & Attendance Policy. 
    The processed data is returned in JSON format.
    Returns:
        JSON: Processed data related to the query about Webkorps.
    """

    try:
        data = request.get_json()
        query = data['query']
        processed_data = process_query('webkorps_data.txt', query)

        return jsonify(processed_data)
    except Exception as e:
        logger.exception("An error occurred in /webkorps_query endpoint.")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/summary', methods=['POST'])
def summary():
    """
    Generates a summary based on the input query and source.
    This function handles POST requests containing a JSON payload with 'query' and 'source' fields. It generates a summary by searching for relevant documents related to the provided source and answering the query using an AI-powered QA model. 
    The result is returned in JSON format.
    Returns:
        JSON: Summary generated based on the input query and source.
    """

    data = request.get_json()
    query = data['query']
    source = data['source']


    embeddings = OpenAIEmbeddings()  
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify(result)
