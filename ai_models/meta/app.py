from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
import os
import logging
from .llama_openai import app as llm_openais_path
# from .llama_2 import app as llama2_path

app = Blueprint('meta_model', __name__)

app.register_blueprint(llm_openais_path.app, url_prefix='/llama_openai')
# app.register_blueprint(llama2_path.app, url_prefix='/llama2')