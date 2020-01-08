from flask import Flask, jsonify, make_response, request, abort, redirect, send_file, render_template
import logging
import base64
import os
import cv2

import sys
localpath = sys.path[0].replace("/src","")
sys.path.append(localpath)

from src.emotion_recognition import process_image


app = Flask(__name__)

@app.route('/')
def index():
    print("Welcome to emotion recognition system")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
#Only used for png file
def predict():
    print("Starting predict emotion")
    json_data = request.get_json()
    image_data = base64.b64decode(json_data["image"])

    print("Starting procecss image")
    (output_image, emotion_result) = process_image(image_data)

    print("Writing temp image to disk")
    image_filename = str(json_data["id"]) + ".png"
    cv2.imwrite(image_filename, output_image)

    print("Encoding output image to base64")
    with open(image_filename, "rb") as img_file:
        output_string = base64.b64encode(img_file.read())
        output_string = output_string.decode("utf-8")

    os.remove(image_filename)

    return make_response(jsonify({"message": "Successfully predict image", "image": output_string, "emotion_result": emotion_result}), 200)

@app.errorhandler(400)
def bad_request(erro):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource no found.'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8084)