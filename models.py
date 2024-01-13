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
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict(model, stored_image):
    preds = model.predict(stored_image)
    return preds


def decode_results(preds, top=3):
    results = decode_predictions(preds, top=top)[0]
    return results
