from flask import Flask, request, jsonify, g
from .helper import *
from dotenv import load_dotenv
from flask_cors import CORS 
import os
import logging
from flask import Blueprint
load_dotenv()


open_ai = Blueprint('open_ai_model', __name__)

from flask_restx import Api, Resource, fields
api = Api(open_ai)

# Define the request body model
query_model = api.model('Query', {
    'query': fields.String(required=True, description='The query string')
})

query_model_source = api.model('Source', {
    'query': fields.String(required=True, description='The query string'),
    'source': fields.String(required=True, description='The query string'),

})
query_model_form = api.model('Form', {
    'query': fields.String(required=True, description='The query string'),
    'source': fields.String(required=True, description='The query string'),
    'type': fields.String(required=True, description='The query string'),

})

# Configure logging
logging.basicConfig(filename='../../logs/open_ai_model.log', level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class HelloWorld(Resource):
    def get(self):
        return 'OK', 200



class cryoport(Resource):
    @api.expect(query_model)
    def post(self):
        """_Processes a query related to Cryoport information.
            This endpoint is designed to handle POST requests and processes queries regarding Cryoport. Upon receiving a valid JSON payload containing a 'query' field, the endpoint extracts the query and processes it to retrieve relevant information about Cryoport and its uses.
            Returns:
            JSON: Processed data related to the query, typically information about Cryoport and its uses.
            The processed data is then returned in JSON format, providing insights and details based on the query submitted to the endpoint.
        """

        try:
            data = request.get_json()
            query = data['query']

            if not query:
                raise ValueError("Please enter a valid input.")
            
            processed_data = process_query('cryoport_text.txt', query)

            return jsonify(processed_data)
        except Exception as e:
            logger.exception("An error occurred in /cryoport endpoint."+str(e))
            return jsonify({'Error': str(e)}), 500



# @open_ai.route('/real-estate/text-generation', methods=['POST'])
class realEstateQuery(Resource):
    @api.expect(query_model)
    def post(self):
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

            if not query:
                raise ValueError("Please enter a valid input.")
            
            processed_data = process_query('estate.txt', query)

            return jsonify(processed_data)
        except Exception as e:
            logger.exception("An error occurred in /realEstateQuery endpoint.")
            return jsonify({'Error': str(e)}), 500




class webkorps_query(Resource):
    @api.expect(query_model)
    def post(self):
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
            if not query:
                raise ValueError("Please enter a valid input.")
            
            processed_data = process_query('webkorps_data.txt', query)

            return jsonify(processed_data)
        except Exception as e:
            logger.exception("An error occurred in /webkorps_query endpoint.")
            return jsonify({'Error': str(e)}), 500


class assetpanda(Resource):
    @api.expect(query_model)
    def post(self):
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

            if not query:
                raise ValueError("Please enter a valid input.")
            
            processed_data = process_query('assetpanda.txt', query)

            success_message = {
                                    "status": "Success",
                                    "Response": processed_data
                            }
            return jsonify(success_message)
                    
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)
            }
            return jsonify(failure_response)



class summary(Resource):
    @api.expect(query_model_source)
    def post(self):
        try:
            data = request.get_json()
            query = data['query']
            source = data['source']
            if not query:
                raise ValueError("Please enter a valid input.")

            embeddings = OpenAIEmbeddings()  
            document_search = FAISS.from_texts([source], embeddings)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            docs = document_search.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            success_message = {
                                    "status": "Success",
                                    "Response": result
                            }
            return jsonify(success_message)
                    
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)
            }
            return jsonify(failure_response)


def get_embedding(source,embeddings):
    query = "Which form we need to create just one word answer like <Create the ________ form> in this format,I need to get the response"
    document_search = FAISS.from_texts([source], embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result.replace('\n', '')

# @open_ai.route('/openai/form', methods=['POST'])

class openai_form(Resource):
    @api.expect(query_model_form)
    def post(self):
        """
        Generates a summary based on the input query and source.
        This function handles POST requests containing a JSON payload with 'query' and 'source' fields. It generates a summary by searching for relevant documents related to the provided source and answering the query using an AI-powered QA model. 
        The result is returned in JSON format.
        Returns:
            JSON: Summary generated based on the input query and source.
        """ 
        # Input query
        # import pdb ; pdb.set_trace()
        try:
            embeddings = OpenAIEmbeddings()
            data = request.get_json()
            query = data["query"]
            source = data['source']
            type = data['type']

            if not query:
                raise ValueError("Please enter a valid input.")
            
            query_words = get_embedding(query,embeddings)
            print(query_words) 

            if type == "create":
                query_sub = f"provide me the relevant columns only for {query_words} would be"

            else:
                target_word = query_words.replace("form", "")
                query_sub = f"provide me the relevant columns name in array only for updating {target_word} would be."

            # query_sub = f"provide me the relevant columns name in array only for {query} would be."
            # Fine-tuning rule to ensure all columns are included
            
            
            # query = f"Create the {query_words} form"
            document_search = FAISS.from_texts([source], embeddings)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            docs = document_search.similarity_search(query)
            result = chain.run(input_documents=docs, question=query_sub)


            success_message = {
                                    "status": "Success",
                                    "Response": result
                            }
            return jsonify(success_message)
                    
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)
            }
            return jsonify(failure_response)




api.add_resource(HelloWorld, '/')
api.add_resource(cryoport,'/open_ai_model/cryoport/text-generation') 
api.add_resource(realEstateQuery,'/open_ai_model/real-estate/text-generation')
api.add_resource(webkorps_query,'/open_ai_model/webkorps/text-generation')
api.add_resource(webkorps_query,'/open_ai_model/webkorps/text-generation')


api.add_resource(assetpanda,'/open_ai_model/openai_response')
api.add_resource(summary,'/open_ai_model/openai/source')
api.add_resource(openai_form,'/open_ai_model/openai/form')
