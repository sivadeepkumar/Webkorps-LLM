from flask import request, jsonify, Blueprint
from .llama_openai.routes import routes as llm_openais_path

llama_ai = Blueprint('meta_model', __name__)

llama_ai.register_blueprint(llm_openais_path, url_prefix='/llama_openai')

