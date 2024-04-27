from flask import Flask
from config.app_config import global_cfg
import tf_model.tensorflow_model as tf_model
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json

tf_instance = tf_model.TFModel(
    global_cfg.config['model']['class_names'],
    global_cfg.config['model']['path']
)

# flask API
app = Flask(__name__, static_folder='statics')


@app.route("/welcome")
def hello_world():
    return "<p>This is a application which classify product categories</p>"


@app.route("/",)
def get_index():
    return render_template('index.html')


@app.route("/api/v1/predict/form-data", methods=['POST'])
def predict():
    image = request.files['image']
    try:
        num_classes = int(request.form['num_classes'])
    except:
        num_classes = 1

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(image.filename))
    image.save(file_path)

    print(file_path)

    result, total_time = tf_instance.predict_img(
        file_path, num_classes=num_classes)

    predicts = list()
    for class_name, confidence in result.items():
        item = PredictResponse(class_name, confidence).to_dict()
        predicts.append(item)

    data = {
        "total_time": str(total_time)+"ms",
        "predicts": predicts,
    }

    return jsonify(data)


class PredictResponse():
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

    def to_dict(self):
        return {
            'class_name': self.name,
            'confidence': str(self.confidence)
        }


if __name__ == '__main__':
    app.run(port=global_cfg.config['server']['port'], debug=True)
