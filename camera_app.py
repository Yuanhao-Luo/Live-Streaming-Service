from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import cv2
import datetime, time
import os, sys
import numpy as np
import face_recognition
import math
from threading import Thread, Timer

global capture,rec_frame, grey, switch, neg, face, rec, out, name, images, classNames, encodeListKnown, camera, path, myList, sin_frame, det_face, match_per
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
det_face=""
match_per=0
name = "Unknown"

path="shots"
images=[]
classNames=[]
encodeListKnown=[]
camera = cv2.VideoCapture(0)
myList = []



def findEncodings(images):
    encodeList =[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)
        encodeList.append(encode[0])
    return encodeList

def load_images():
    global images, classNames, encodeListKnown, face, myList

    face = 0

    images = []
    classNames = []
    encodeListKnown = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # get name only

    encodeListKnown = findEncodings(images)
    print("encoding complete!")
    face = 1

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
socketioApp = SocketIO(app)

#load images and encodings
load_images()

camera = cv2.VideoCapture(0)



def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
            return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass

def recognize_face(frame):
    global face, name, match_per, det_face

    frame = cv2.flip(frame, 1)

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    # image changes to RGb for the face_recognition library
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesInCurrentFrame = face_recognition.face_locations(imgS)  # multiple faces
    if len(facesInCurrentFrame) == 1 and det_face != "" and det_face != "Unknown":
        y1, x2, y2, x1 = facesInCurrentFrame[0]
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, det_face + " " + str(match_per) + "%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)
        frame = cv2.flip(frame, 1)
        return frame
    elif len(facesInCurrentFrame) == 0:
        det_face = ""

    encodingsCurrentFrame = face_recognition.face_encodings(imgS, facesInCurrentFrame)

    # Loop through the face encodings and compare them
    for encodeFace, faceLocation in zip(encodingsCurrentFrame, facesInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        # Match index is set to the lowest distance that is the most accurate.
        matchIndex = np.argmin(faceDist)
        # Check it index exists
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            matchPerc = round(face_distance_to_conf(faceDist[matchIndex]) * 100)
            det_face = name
            match_per = matchPerc

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name + " " + str(matchPerc) + "%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)
        else:
            det_face = "Unknown"
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    frame = cv2.flip(frame, 1)

    return frame

def gen_per_frame():
    global out, capture,rec_frame, sin_frame
    success, frame = camera.read()
    if success:
        if (face):
            frame = recognize_face(frame)
        if (grey):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (neg):
            frame = cv2.bitwise_not(frame)
        if (capture):
            capture = 0
            now = datetime.datetime.now()
            if name == "":
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
            else:
                p = os.path.sep.join(['shots', "{}.png".format(name)])
            cv2.imwrite(p, frame)
            load_images()

        if (rec):
            rec_frame = frame
            frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                4)
            frame = cv2.flip(frame, 1)

        sin_frame = frame

 

def gen_frames():  # generate frame by frame from camera
    global sin_frame
    while True:
        thread = Thread(target=gen_per_frame)
        thread.start()
        time.sleep(0.1)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(sin_frame, 1))
            sin_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + sin_frame + b'\r\n')
        except Exception as e:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('capture') == 'Capture':
            global capture
            global name
            name=request.form.get('name')
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Recognization':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
