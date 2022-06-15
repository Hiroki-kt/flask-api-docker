from flask import Flask, request, jsonify, make_response
from mediapipe_pose import MediaPipePose

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello flask !!"


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
    # app.logger.warning(params)
    response = {}
    mpp = MediaPipePose(dryrun=True)
    # image, results = mpp.get_born(params.get('image_fpath'))
    image, results = mpp.get_born('./test-image1.jpg')
    # app.logger.warning(results)
    mpp.write_born(image, results, None, landmark_on=True)
    crop_result = mpp.crop_landmark(
        image, results, None, random_crop=True)
    # if crop_result is not None:
    if 'param' in params:
        response.setdefault('res', results)
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
