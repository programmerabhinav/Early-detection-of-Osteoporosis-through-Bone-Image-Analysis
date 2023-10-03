from re import L
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model 

# img = 'img.png'
# img = cv2.imread(img)

model = load_model('Osteoporosis_73.h5')

app = Flask(__name__)

@app.route('/',methods = ['POST','GET'])
def main():
    if request.method == 'POST':
        imagefile = request.files.get('imagefile', '')
        imagefile.save("static/output.jpg")
        img = cv2.imread('static/output.jpg')
        img = np.array([cv2.resize(img,(100,100))])
        r = {"0":'Osteoporosis', 
        "1":'Normal'}
        rc = {"0":'red','1':"green"}
        output = np.argmax(model.predict(img), axis=1)
        color_data = {'color':rc[str(output[0])]}
        data = {"output":r[str(output[0])]}
        return render_template('upload.html', data = data, color_data = color_data)
    elif request.method == 'GET':
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
