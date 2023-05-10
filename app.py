from sre_parse import CATEGORIES
from flask import Flask, render_template, request, jsonify 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import decode_predictions 
from keras.applications.vgg16 import VGG16 
from keras.models import load_model
from werkzeug.utils import secure_filename
import base64
import os
import numpy as np
import tensorflow.keras.applications.xception as xception
 


IMAGE_WIDTH = 320    
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

categories = {1: 'paper', 2: 'cardboard', 3: 'plastic', 4: 'metal', 5: 'trash', 6: 'battery',
              7: 'shoes', 8: 'clothes', 9: 'green-glass', 10: 'brown-glass', 11: 'white-glass',
              12: 'biological'}

app = Flask(__name__)
model = load_model('/users/fadihusary/Desktop/gc.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/", methods=["POST"])
@app.route("/", methods=["POST"])
def classify_garbage():
    imagefile= request.files['imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size = (IMAGE_WIDTH,IMAGE_HEIGHT))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    preds = model.predict(image)
    predicted_class_index = np.argmax(preds, axis=1)[0]

    predicted_class_name = categories[predicted_class_index + 1]



    classification = f'{predicted_class_name} ({preds[0][predicted_class_index] * 100:.2f}%)'


    return render_template('index.html', prediction = classification)

if __name__=='__main__':
    app.run(port=3000, debug=True)
