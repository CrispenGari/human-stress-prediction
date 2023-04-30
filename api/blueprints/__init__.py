from flask import Blueprint, make_response, jsonify, request
from api.model import device, predict_stress, hsp_model
import time
blueprint = Blueprint("blueprint", __name__)

@blueprint.route('/v1/predict-stress', methods=["POST"])
def predict_stress():
    start = time.time()
    data = {"success": False, "time": 0}
    if request.method == "POST":
        try:
            if request.is_json:
                json = request.get_json(force=True)
                if json.get("text"):
                    res = predict_stress(hsp_model, json.get("text"), device)
                    data = {
                        'success': True,
                        'prediction': res.to_json(),
                    }
                else:
                    data['error']  = "you should pass the 'text' in your json body while making this request."
            else:
                raise Exception("the is no json data in your request.")
        except Exception as e:
            print(e)
            data['error'] = 'something went wrong on the server'
    else:
        data['error']  = "the request method should be post only."
    end = time.time()
    data["time"] = end - start
    return make_response(jsonify(data)), 200