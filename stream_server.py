from flask import Flask, Response
import cv2
from frame_buffer import frame_buffer

app = Flask(__name__)

def generate_mjpeg():
    while True:
        frame = frame_buffer.obter(timeout=1.0)
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')