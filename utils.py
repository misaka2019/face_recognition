from flask import jsonify
import base64
import numpy as np
import cv2


def check_recognition_param(pass_param, post_json):
    data = {}
    for key in post_json.keys():
        if key not in pass_param:
            msg = "get unexpected params '%s'" % key
            return False, msg, None

    for param in pass_param:
        if param not in post_json:
            msg = "missing required parameter '%s'" % param
            return False, msg, None
        elif post_json[param] is None or post_json[param] is "":
            msg = "the parameter '%s' is empty" % param
            return False, msg, None
        elif param == 'image':
            image_base64 = post_json["image"]
            try:
                image = base64_to_np(image_base64)
                data['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                msg = "parameter 'image' error"
                return False, msg, None
        else:
            data[param] = post_json[param]

    return True, None, data


def check_insert_param(pass_param, post_json):
    data = {}
    for param in pass_param:
        if param not in post_json:
            msg = "missing required parameter '%s'" % param
            return False, msg, None
        elif post_json[param] is None or post_json[param] is "":
            msg = "the parameter '%s' is empty" % param
            return False, msg, None
        else:
            data[param] = post_json[param]

    return True, None, data


def check_delete_param(pass_param, post_json):
    data = {}
    if post_json is None:
        msg = "The JSON data obtained is none"
        return False, msg, None
    for param in pass_param:
        if param not in post_json:
            msg = "missing required parameter '%s'" % param
            return False, msg, None
        elif post_json[param] is None or post_json[param] is "":
            msg = "the parameter '%s' is empty" % param
            return False, msg, None
        else:
            data[param] = post_json[param]
            if not isinstance(data[param], list):
                msg = "'mid' param isn't array "
                return False, msg, None
            return True, None, data


def result(isSuccess, status_code, msg, data=None):
    if data is None:
        return jsonify({"isSuccess": isSuccess,
                        "code": status_code,
                        "message": msg
                        })
    else:
        return jsonify({"isSuccess": isSuccess,
                        "code": status_code,
                        "message": msg,
                        'data': data
                        })


# 将base64格式数据转换为NumPy数组
def base64_to_np(img_base64):
    img_data = base64.b64decode(img_base64)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


# 将NumPy数组转换为base64格式数据
def np_to_base64(img_np):
    if img_np is None:
        return img_np
    if len(img_np) == 0:
        return img_np

    retval, buffer = cv2.imencode('.jpg', img_np)
    img_base64 = base64.b64encode(buffer)
    img_base64 = img_base64.decode()

    return img_base64
