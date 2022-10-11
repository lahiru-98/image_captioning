import pickle as pkl
import pandas as pd
from flask import Flask , request
from pathlib import Path
import os 
from flask import send_from_directory     
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm.notebook import tqdm
from PIL import Image


app = Flask(__name__)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

featureModel = VGG16()
featureModel = Model(inputs=featureModel.inputs, outputs=featureModel.layers[-2].output)


InputDatafeatures = {}
#directory = os.path.join(TEST_DIR, 'InputImages')

for img_name in tqdm(os.listdir(directory)):
        # load the image from file
        img_path = 'sample.png' #directory + '/' + img_name
        print(img_path)
        image = load_img(img_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for vgg
        image = preprocess_input(image)
        # extract features
        imagefeature = featureModel.predict(image, verbose=0)
        # get image ID
        image_id = img_name.split('.')[0]
        # store feature
        InputDatafeatures[image_id] = imagefeature

def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return "Model loaded"

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
       
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequenc
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(TEST_DIR, "InputImages", image_name)
    image = Image.open(img_path)
    # predict the caption
    y_pred = predict_caption(model, InputDatafeatures[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    y_pred = y_pred.replace("startseq", "")
    y_pred = y_pred.replace("endseq", "")
    print(y_pred)
   



@app.route('/')
def index():
    text = load_model()
    return "Your App is Working!!!" + text

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')




if __name__ == "__main__":
    app.run()
    
    
    
    