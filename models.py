import numpy as np
from tensorflow import keras
from keras.applications.resnet import ResNet50
import keras.utils as image
from keras.applications.resnet import preprocess_input, decode_predictions


def load_model():
    model = ResNet50(weights='imagenet')
    # Optionally, add model compilation details here
    return model


def preprocess_image(filename):
    # loads the image and resizes it to (224, 224). ResNet input requirement
    img = image.load_img(filename, target_size=(224, 224))
    # converts the image to a NumPy
    x = image.img_to_array(img)
    # adds a batch dimension as most models expect a 4D tensor for prediction
    x = np.expand_dims(x, axis=0)
    # performs specific transformations for the model
    x = preprocess_input(x)
    return x


def predict(model, stored_image):
    # prediction using the loaded model
    preds = model.predict(stored_image)
    return preds


def decode_results(preds, top=3):
    # decodes the prediction to extract meaningful labels or scores.
    results = decode_predictions(preds, top=top)[0]
    return results
