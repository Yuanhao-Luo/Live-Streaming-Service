import eventlet
from camera_app import app

import socketio
from waitress import serve
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)

if __name__ == '__main__':
    print("Started at http://127.0.0.1:8080 or http://" + IPAddr + ":8080  5 clients max")
    serve(appServer, host='0.0.0.0', port=8080, url_scheme='http', threads=5)

