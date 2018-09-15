# unused in project
from flask import Flask
from flask_cors import CORS

speech_score = 50
expression_score = 50
hands_score = 50

app = Flask(__name__)
CORS(app)   # allow cross domain


@app.route("/expression")
def expression():
    return '{"val":' + str(expression_score) + "}"


@app.route("/hands")
def hands():
    foo()
    return '{"val":' + str(hands_score) + "}"


@app.route("/speech")
def speech():
    foo()
    return '{"val":' + str(speech_score) + "}"
