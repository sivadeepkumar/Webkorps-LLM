import boto3
from flask import Flask, render_template, request,jsonify
from flask import Blueprint
from .bedrock_kb_agent import BedrockKBAgent

app = Flask(__name__)
app = Blueprint('amazon_model', __name__)
kb = BedrockKBAgent()
client = boto3.client('comprehend', region_name='us-east-1')



@app.route('/bedrock', methods=['POST'])
def bedrock_tech():
    """
        Handles queries related to a specific knowledge base using BedrockKBAgent.

        This endpoint takes a POST request with a JSON payload containing the 'query' key representing the user's query. It uses BedrockKBAgent to retrieve information from a specific knowledge base identified by the 'kb_id' variable and returns the retrieval results in JSON format.

        Parameters:
            None (Request payload contains the 'query' key with the user's query string).

        Returns:
            JSON: A JSON response containing the retrieval results from the specified knowledge base.

        Raises:
            None (Any errors in retrieving data from the knowledge base are handled internally within BedrockKBAgent).
    """

    data = request.get_json()
    query = data.get('query')
    kb_id = "QRJWFQFERS"
    response = kb.retrieve_from_kb(kb_id, query)
    return response['retrievalResults'] # [0]['content']


@app.route('/sample', methods=['POST'])
def query_model():
    """
        Queries a knowledge base using Amazon Comprehend and returns the response.

        This endpoint takes a POST request with a JSON payload containing the 'query' key representing the user's query. It invokes the Amazon Comprehend model to query a knowledge base specified by the 'knowledge_base_id' variable and returns the response in JSON format.

        Parameters:
            None (Request payload contains the 'query' key with the user's query string).

        Returns:
            JSON: A JSON response containing the response from the Amazon Comprehend model.

        Raises:
            None (Any errors in invoking the Amazon Comprehend model are handled internally).
    """


    # Get the query from the request data
    query_data = request.get_json()
    query = query_data.get('query', '')

    # Specify the knowledge base ID
    knowledge_base_id = 'QRJWFQFERS'

    # Invoke the model
    response = client.invoke_knowledge_base(
        KnowledgeBaseId=knowledge_base_id,
        Query=query
    )

    # Process the response
    if 'Response' in response:
        # Return the response
        return jsonify({'response': response['Response']})
    else:
        return jsonify({'error': 'No response received'})
    