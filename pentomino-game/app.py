import json
import os
from cgitb import html
import numpy as np
import cv2
import base64

import game.board as board
from flask import request, Flask, render_template, jsonify, make_response

app = Flask(__name__)
app.debug = True
if os.environ.get("IS_DOCKER", "False") == "True":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_prefix=1)

logger = app.logger


@app.before_first_request
def setup():

    logger.info("Setting up models, this may take some time.")

    logger.info("Setup finished")


@app.route("/")
def index() -> html:
    return render_template('index.html')


@app.route('/update_images', methods=["GET"])
def get_image():
    global stream

    logger.debug("Analyzing image")

    image = cv2.imread("test.png")

    if image is None:
        return jsonify({'status': False})

    img_base64 = base64.b64encode(cv2.imencode('.png', image)[1])

    return jsonify({'status': True, 'data': img_base64.decode("utf-8")})


if __name__ == '__main__':
    ip = '172.26.44.176'
    ip0 = '0.0.0.0'
    ip1 = '127.0.0.1'
    global stream
    stream = board.VideoStream().start()
    #app.run(ssl_context='adhoc')
    app.run(host=ip0, debug=True, port=5000)
