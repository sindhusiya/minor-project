from flask import Flask, render_template, Response,redirect
import cv2
import threading

app = Flask(__name__)

# Set up video capture and YOLO model
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.6
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ["Bread", "CardBoard", "Metal Can", "Plastic Bag"]

wei = "C:/Users/91905/OneDrive/Desktop/spyder/yolov4.cfg"
cf = "C:/Users/91905/OneDrive/Desktop/spyder/yolov4.weights"
net = cv2.dnn.readNet(wei, cf)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv2.VideoCapture(0)
is_detecting = False

def detect_objects():
    global is_detecting
    global cap
    
    while is_detecting:
        check, frame = cap.read()
        if not check:
            break

        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for (classid, score, box) in zip(classes, scores, boxes):
            classid=int(classid)
            label = "%s : %f" % (class_names[classid], score)
            cv2.rectangle(frame, box, COLORS[classid], 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classid], 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('deek.html')

@app.route('/start', methods=['POST'])
def start_detection():
    global is_detecting
    global cap
    cap = cv2.VideoCapture(0)
    is_detecting = True
    return redirect("/")

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_detecting
    global cap
    if is_detecting:
        is_detecting = False
        cap.release()
        cv2.destroyAllWindows()
        return redirect("/")
    return redirect("/")
    

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
