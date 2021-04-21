from flask import Blueprint
from flask import request

from utils import *
from face_utils.face_recognition import ImageFaceRecognition

face_info = Blueprint('face_app', __name__)

require_insert_param = ['replace', 'uid', 'image']
require_recognition_param = ['image']
require_delete_param = ['mid', 'uid']

# 图片人脸识别模型
face_reco_model = ImageFaceRecognition()


@face_info.route("/image/faceInsert", methods=['POST'])
def image_entryInformation():
    isSuccess = False
    status_code = 400

    post_json = request.get_json()
    if post_json is None:
        msg = "The JSON data obtained is none"
        return result(isSuccess, status_code, msg)
    isSuccess, msg, param = check_insert_param(require_insert_param, post_json)
    if isSuccess:
        isSuccess, msg, status_code, image = face_reco_model.insert_face(param['image'], param['uid'],
                                                                         param['replace'])

        return result(isSuccess, status_code, msg)
    else:
        return result(isSuccess, status_code, msg)


@face_info.route("/image/faceRecognition", methods=['POST'])
def image_faceRecognition():
    isSuccess = False
    status_code = 400
    uids, unrecognizedImage, recognizedImage, mids, markImage_base64 = [], [], [], [], None
    data = {
        "uids": uids,
        "markImage": markImage_base64,
    }
    post_json = request.get_json()
    if post_json is None:
        msg = "The JSON data obtained is none"
        return result(isSuccess, status_code, msg, data)
    isSuccess, msg, param = check_recognition_param(require_recognition_param, post_json)
    if isSuccess:
        status_code, isSuccess, msg, unrecognizedImage, uids, markImage, mids, recognizedImage = face_reco_model.reco_face(
            param['image'])
        data = {
            "uids": uids,
            "markImage": markImage
        }

        return result(isSuccess, status_code, msg, data)
    else:
        return result(isSuccess, status_code, msg, data)


@face_info.route('/image/faceDelete', methods=['POST'])
def image_faceDelete():
    status_code = 500
    post_json = request.get_json()
    isSuccess, msg, param = check_delete_param(require_delete_param, post_json)

    if isSuccess:
        isSuccess, msg, status_code = face_reco_model.face_delete(post_json['mid'], post_json['uid'])

        return result(isSuccess, status_code, msg)
    else:
        return result(isSuccess, status_code, msg)
