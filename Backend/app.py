import os
from flask import Flask, render_template, request
from flask_restful import  Api
from flask_cors import CORS
from main import generate_mcq_questions
app = Flask(__name__)

CORS(app)

api = Api(app)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/generate-mcq', methods=['POST'])
def generate_mcq():

    try:
        print("hello")
        request.get_json(force=True)
        print(request.json['text'])
        text = {"input_text": str(request.json['text'])}
        print(text)
        mcq = generate_mcq_questions({"input_text": str(request.json['text'])})
        print(mcq)
        ## return as json
        return mcq, 200


    except Exception as e:
        print(e)
        return "Some thing went wrong " + str(e), 400



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)





