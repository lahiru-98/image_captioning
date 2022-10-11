import pickle as pkl
import pandas as pd
from flask import Flask , request
from pathlib import Path
import tempfile 

app = Flask(__name__)


@app.route('/')
def index():
    return "Your App is Working!!!"



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
    
    
    
    