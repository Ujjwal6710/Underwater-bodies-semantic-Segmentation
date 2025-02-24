import os
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image

app = Flask(__name__)

IMAGE_SIZE = 256
MODEL_PATH = "saved_models/deeplab.pkl"
RESULT_DIR = "result_images"
os.makedirs(RESULT_DIR, exist_ok=True)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

def load_image(img_path):
    image = load_img(img_path)
    image = img_to_array(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))        
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return tf.expand_dims(image, axis=0)



@app.route("/store", methods=["POST"])
def create_item():
    data = request.get_json()
    image_name = data["image_name"]
    
    image_path = os.path.join("saved_images", image_name)

    # Load and process the image
    image = load_image(image_path)

    # Predict the result using the model
    result = model.predict(image)

    result_image_path = os.path.join(RESULT_DIR, image_name)
    result_img = result[0, :, :, 0] * 255 
    result_img = tf.cast(result_img, tf.uint8).numpy()
    
    # Save as an image
    result_image = Image.fromarray(result_img)
    result_image.save(result_image_path)

    return jsonify({"result_path": result_image_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
