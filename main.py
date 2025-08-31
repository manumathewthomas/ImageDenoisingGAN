from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from io import BytesIO
from test import *
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/denoisify', methods=['GET', 'POST'])
def denoisify():
    if request.method == "POST":
        inputImg = request.files['file']
        # Convert the file object to a numpy array using OpenCV
        input_array = cv2.imdecode(np.frombuffer(inputImg.read(), np.uint8), cv2.IMREAD_COLOR)
        input_array = cv2.cvtColor(input_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        outputImg = denoise(input_array)
        print(f"Out image: {outputImg}")
        
        # Save the output image using OpenCV
        output_image = (outputImg * 255).astype(np.uint8)  # Scale to 0-255 range
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving
        cv2.imwrite('static/output.png', output_image)

        return jsonify(result="Success")

if __name__=="__main__":
    app.run(host="0.0.0.0",port="80")
