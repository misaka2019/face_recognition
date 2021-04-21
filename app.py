from flask import Flask
from flask_httpauth import HTTPBasicAuth

from setting import setting
from utils import *
from face_utils.view import face_info


config = setting()

"""
定义api
"""

app = Flask(__name__)
auth = HTTPBasicAuth()


# 人脸识别
app.register_blueprint(face_info, url_prefix="/")


@auth.error_handler
def unauthorized():
    return jsonify({"isSuccess": False,
                    "code": 401,
                    "message": "Unauthorized access", })


@app.errorhandler(404)
def not_service(error):
    return jsonify({"isSuccess": False,
                    "code": 404,
                    "message": "NOT FOUND", })


@app.errorhandler(400)
def input_err(error):
    return jsonify({"isSuccess": False,
                    "code": 400,
                    "message": "Invalid data!", })


@app.errorhandler(500)
def intern_err(error):
    return jsonify({"isSuccess": False,
                    "code": 500,
                    "message": "Internal error!", })


if __name__ == "__main__":
    app.run(host=config.host, port=config.app_port, debug=False)
