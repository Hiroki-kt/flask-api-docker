from flask import Flask, request, jsonify, make_response

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
