from flask import Flask, request, jsonify, make_response
from mediapipe_pose import MediaPipePose

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello flask"


@app.route("/api", methods=["POST"])
def postHoge():
    params = request.json
    print(request)
    response = {}
    if 'param' in params:
        response.setdefault('res', f'param is: {params.get("param")}')
    return make_response(jsonify(response))

@app.route("/api/mediapipe", methods=["POST"])
def getPost():
    params = request.json
    # print(params)
    response = {}
    mpp = MediaPipePose()
    image, results = mpp.get_born(params.get('image_fpath'))
    crop_result = mpp.crop_landmark(
        image, results, None, resize_size=(112, 112), random_crop=True)
    if crop_result is not None:
        if 'param' in params:
            response.setdefault('res', crop_result)
        return make_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
