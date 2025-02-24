import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, Response,redirect
import threading
import csv
import json

app = Flask(__name__)

# Define the path to the directory containing student images
path = 'student_images'

# Global variables for video capture and recognition
cap = None
is_recognizing = False
encoded_face_train = []
images=[]
classNames = []
# Create a CSV file for attendance recording
def createCSV():
    now = datetime.now()
    date = now.strftime('%d-%B-%Y')
    filename = date + "-Attendance.csv"
    temp = open(filename, 'a+')
    temp.close()
    return filename

# Mark attendance in the CSV file
def markAttendance(name, filename):
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}\n')

# Load student images and associated class names
def loadStudentImages():
    global encoded_face_train
    global images
    global classNames
    images = []
    classNames = []
    mylist = os.listdir(path)
    for cl in mylist:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encoded_face_train = findEncodings(images)

# Create a list of face encodings for the loaded images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
 
# Video streaming generator function
def generate():
    global cap, is_recognizing
    while is_recognizing:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                confidence = (1 - faceDist[matchIndex]) * 100
                # print(confidence)
                if confidence > 30:
                    name = classNames[matchIndex].upper().lower()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name, filename)

        ret, frame = cv2.imencode('.jpg', img)
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('minor.html')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global cap, is_recognizing
    if not is_recognizing:
        cap = cv2.VideoCapture(0)
        is_recognizing = True
        return redirect("/")
    return redirect("/")    
        
    
 
@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global cap, is_recognizing
    if is_recognizing:
        is_recognizing = False
        cap.release()
        cv2.destroyAllWindows()
        return redirect("/")
    return redirect("/")

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_report')
def get_report():
    # Read data from the CSV file
    report_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            report_data.append(row)

    return json.dumps(report_data)


if __name__ == '__main__':
    loadStudentImages()
    filename = createCSV()
    app.run(debug=True)
