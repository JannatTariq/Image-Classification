from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, static_folder='static')
model = tf.keras.models.load_model('mnist1.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Open the image using Pillow
        pil_image = Image.open(file).convert('L')
        # Resize the image to 28x28
        pil_image = pil_image.resize((28, 28))
        # Convert the image to a numpy array
        np_image = np.array(pil_image)
        # Reshape the image to a 1D array with 784 elements
        np_image = np_image.reshape(1, 784)
        # Normalize the pixel values
        np_image = np_image / 255.0
        # Use the model to predict the class of the input image
        prediction = model.predict(np_image)
        # Get the predicted class label
        predicted_class = np.argmax(prediction[0])
        return render_template('index.html', prediction=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
