from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import Inference
import numpy as np
import cv2, os


# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default
infer = Inference('./config.yaml')
cfg = infer.infer_cfg
model_api = infer.get_model_api()


# default route
@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


# API route
@app.route('/api', methods=['POST', 'GET'])
def api():
    file_path = request.args['filepath']
    assert os.path.exists(file_path), file_path
    image = cv2.imread(file_path)
    image = image[:, :, :3]  # remove alpha
    image = image[:, :, ::-1]  # BGR -> RGB
    h, w = cfg.network_input_shape
    patches = []
    for crop in cfg.frame_crops:
        y1, x1, y2, x2 = crop
        y1, y2 = int(h * y1), int(h * y2)
        x1, x2 = int(h * x1), int(h * x2)
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, (w, h),
                           interpolation=cv2.INTER_AREA)
        patches.append(patch)
    # r = request
    # # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    # # decode image
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    bboxes, top_classes, top_probs = model_api(patches)
    output_data = []
    for bbox, top_classes_, top_probs_ in zip(
        bboxes, top_classes, top_probs):
        if top_probs_[0] < 0:
            continue
        if top_classes_[0] == 0:
            continue
        output_data.append({'bbox': bbox.tolist(),
                            'top_classes': top_classes_.tolist(),
                            'top_probs': top_probs_.tolist()})
    response = jsonify(output_data)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
