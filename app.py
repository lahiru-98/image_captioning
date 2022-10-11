import pickle as pkl
import pandas as pd
from flask import Flask , request
from pathlib import Path
import os 
from flask import send_from_directory     
import tensorflow as tf




app = Flask(__name__)

def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return "Model loaded"
    



@app.route('/')
def index():
    text = load_model()
    return "Your App is Working!!!" + text

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# @app.route('/video', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         uploaded_file = request.files['file']
#         with tempfile.TemporaryDirectory() as td:
#             temp_filename = Path(td) / 'uploaded_video'
#             uploaded_file.save(temp_filename)

#             res = extractFeatures(temp_filename)
#             return str(res)
#     else:
#         return "Something Went Wrong"



if __name__ == "__main__":
    app.run()
    
    
    
    