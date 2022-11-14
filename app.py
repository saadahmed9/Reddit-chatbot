from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from flask_cors import CORS
from chat import possible_soln

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)
CORS(app)


@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    text2 = request.get_json().get("topic")
    print(text2, request.get_json())
    response = possible_soln(text, text2)
    # print(response)
    if 'sim_msg' in response:
        message = {"question": response['question'] ,"answer": response['message_response'], "query_embed": response['query'],'sim_msg': response['sim_msg'], "cos_embed": response['cos_embed'], "preds": response['preds'] }
    else:
        message = {"question": response['question'],"answer": response['message_response'], "query_embed": response['query'],"preds": response['preds']}

    m1 = json.dumps(message, cls=NpEncoder)
    print(message)
    return m1

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
