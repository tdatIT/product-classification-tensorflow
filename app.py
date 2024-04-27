from flask import Flask
from config.app_config import global_cfg
import tf_model.tensorflow_model as tf_model
from flask import Flask, render_template

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


if __name__ == '__main__':
    app.run(port=global_cfg.config['server']['port'], debug=True)
