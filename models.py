from tensorflow import keras
from keras.applications.resnet import ResNet50

def load_model():
  model = ResNet50(weights='imagenet')
  # Optionally, add model compilation details here
  return model
