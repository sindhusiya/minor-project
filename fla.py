from distutils.log import debug
from flask import Flask, render_template,request,jsonify
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle





app = Flask(__name__)
path = 'student_images'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

def createCSV():
    now = datetime.now()
    date = now.strftime('%d-%B-%Y')
    filename=date+"-Attendance.csv"
    print(filename)
    temp=open(filename,'a+')
    temp.close()
    return filename


def markAttendance(name,filename):
   
    with open(filename,'r+') as f:
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

# Initialize a sample variable
sample_variable ="OFF"


filename=createCSV()
def auto_ONOFF():
    while True:

        if sample_variable=="ON":        
            cap  = cv2.VideoCapture(0)
            success, img = cap.read()
            imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faces_in_frame = face_recognition.face_locations(imgS)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
            for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)
                print(matchIndex)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()
                    y1,x2,y2,x1 = faceloc
                    # since we scaled down by 4 times
                    y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    markAttendance(name,filename)
                    # studentset.add(name)
            cv2.imshow('webcam', img)
        else:
            # cv2.destroyWindow("I2")
            cap.release()
            cv2.destroyAllWindows()
            break

@app.route('/')
def hello():
   return render_template("temp.html") 

@app.after_request
def after_request(response):
    # You can perform any post-request tasks here
    print("Request finished processing!")
    print(request.path)
    if(request.path=="/variable"):
        if(sample_variable=="ON"):
            auto_ONOFF()    
    return response
# Create a route to get and update the variable
@app.route('/variable', methods=['GET', 'POST'])
def handle_onoff():
    global sample_variable
    # print(sample_variable)
    if request.method == 'GET':
        return jsonify({'variable': sample_variable})
    elif request.method == 'POST':
        new_value = request.json.get('new_value')
        if new_value:
            sample_variable = new_value
            return jsonify({'message':'Variable updated successfully'})
        return jsonify({'message': 'New value not provided'})

if __name__ == '__main__':
    app.run(debug=True)
