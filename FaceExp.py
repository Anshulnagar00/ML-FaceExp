from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import json
#later
import cv2
from gevent.pywsgi import WSGIServer

# image 2 datauri

from PIL import Image
import io
import base64

# mtcnn classifier
from numpy import asarray
from mtcnn.mtcnn import MTCNN

import time

# firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage



# Fetch the service account key JSON file contents
cred = credentials.Certificate('firebase.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceexp-26f12.firebaseio.com/'
})


# As an admin, the app has access to read and write all data, regradless of Security Rules
ref = db.reference('restricted_access/secret_document')


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(THIS_FOLDER,'FaceExp-V2.0_model (6).h5')
UPLOAD_FOLDER = os.path.join(THIS_FOLDER,'static')
TEMPLATES_FOLDER = 'templates'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}



# Define a flask app
app = Flask(__name__,template_folder=TEMPLATES_FOLDER,static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


FaceExp_model = tf.keras.models.load_model(MODEL_PATH)

# Haarcascade model for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_faces(image_path,size=(160,160)):
    results=[]
    img = cv2.imread(image_path)
    face_img = img.copy()
    temp_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(temp_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        temp_img = face_img[y:y+h,x:x+w]
        temp_img = cv2.resize(temp_img,size)
        results.append(temp_img)
    return results


def firebase_upload(file_path):
  # load image from file
  image = Image.open(file_path)
  # convert to RGB, if needed
  image = image.convert('RGB')
  # convert to array
  pixels = asarray(image)
  # create the detector, using default weights
  db.reference('images').child(time.ctime()).set(image_2_dataURI(pixels))
    

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
  # load image from file
  image = Image.open(filename)
  # convert to RGB, if needed
  image = image.convert('RGB')
  # convert to array
  pixels = asarray(image)
  # create the detector, using default weights
  detector = MTCNN()
  # detect faces in the image
  results = detector.detect_faces(pixels)
  images=[]
  if len(results)==0:
    return images

  # extract the bounding box from the first face

  for img in results:
    x1, y1, width, height = img['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    face_array = cv2.cvtColor(face_array,cv2.COLOR_BGR2RGB)
    images.append(face_array)
  return images




def is_face_present(image_path):
    face = extract_face(image_path)
    if len(face) == 0:
      print('No Faces Detected')
      return False
    else:
      print(str(len(face))+ " Faces Detected")
      return True

def process_batch(images):
  batch=[]
  for image in images:
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray,(48,48))
    img_gray = np.reshape(img_gray,(48,48,1))
    img = np.array(img_gray/255,dtype=np.float32)
    batch.append(img.tolist())
  return batch


def image_2_dataURI(image):
  img = Image.fromarray(image,'RGB')
  rawBytes = io.BytesIO()
  img.save(rawBytes, "PNG")
  rawBytes.seek(0)  # return to the start of the file
  data = "data:image/*;base64,"+base64.b64encode(rawBytes.read()).decode("UTF-8")
  return data

def get_descp_results(img_path):
  res = {}
  if not is_face_present(img_path):
    res['-1']="No faces Detected"
    return res
  else:
    pred=[]
    faces = extract_face(img_path)
    temp = process_batch(faces)
    print(temp)
    pred=FaceExp_model.predict(temp)
    print(pred)

    tags=["angry","disgust","fear","happy","sad","surprise","neutral"]
    for f in range(len(faces)):
      face_img = cv2.resize(faces[f],(120,120))
      face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
      face_img = image_2_dataURI(face_img)

      m={}
      for i in range(len(tags)):
        m[tags[i]]=float("{:.2f}".format(round(pred[f][i], 2)))
        res[str(f)] = {"face":face_img,"predictions":m}

    return json.dumps(res, indent=2)


@app.route('/', methods=['GET'])
def index():
    # Main page
    #f=open('Logs.txt','r')
    #data=f.readline()
    data="khgk"
    return render_template('index.html',urldata=data)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER,secure_filename("IMG_"+time.ctime()+f.filename))
        f.save(file_path)
        firebase_upload(file_path)
        result = get_descp_results(file_path)
        ##########  Final Predictions  ##########
        return result
    return None

@app.route('/about', methods=['GET'])
def about():
  return render_template('About.html')

@app.route('/receivefeedback',methods=['GET','POST'])
def feedback():
  if request.method == 'POST':
    f = request.form.get('feedback')
    data="FeedBack :"+time.ctime()+" -> "+f
    print(data)
    db.reference('Feedbacks').child(time.ctime()).set(data)
    return "Success"

@app.route('/help', methods=['GET'])
def help():
  return render_template('help.html')


