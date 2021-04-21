import cv2
import numpy as np


# 画出人脸框
def draw_face_frame(image, boxes, uids=None):

    if uids is None:
        for box in boxes:
            draw_rectangle(image, box)
    else:
        for uid, box in zip(uids, boxes):
            draw_rectangle(image, box)
            draw_text(image, box, uid)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def draw_rectangle(image, box):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)


def draw_text(image, box, text):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    cv2.putText(image, '{}'.format(text), (box[0]-3, box[1]+1), font, 1, (0, 255, 255), 3)


# 计算两个矩阵之间的欧式距离（公式法，加快运行速度)
def mat_euclidean_distance(A, B, squared=False):
    # A和B由纵向量组成
    A_square = np.sum(np.multiply(A, A), axis=1)
    if A is B:
        B_square = A_square.T
    else:
        B_square = np.sum(np.multiply(B, B), axis=1).T
    distances = np.dot(A, B.T)
    distances *= -2
    distances += A_square
    distances += B_square
    np.maximum(distances, 0, distances)
    if A is B:
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


# 灰度图和高斯滤波
def precess_image(image):
    """
    Graying and GaussianBlur
    :param image: The image matrix,np.array
    :return: The processed image matrix,np.array
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # 高斯滤波
    return gray_image


# 计算帧间差分
def abs_diff(pre_image, curr_image):
    """
    Calculate absolute difference between pre_image and curr_image
    :param pre_image:The image in past frame,np.array
    :param curr_image:The image in current frame,np.array
    :return:
    """
    gray_pre_image = precess_image(pre_image)
    gray_curr_image = precess_image(curr_image)
    diff = cv2.absdiff(gray_pre_image, gray_curr_image)
    res, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #  fixme：这里先写成简单加和的形式
    cnt_diff = np.sum(np.sum(diff))
    return cnt_diff


def exponential_smoothing(alpha, s):
    s_temp = [s[0]]
    # print(s_temp)
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i - 1] + (1 - alpha) * s_temp[i - 1])
    return s_temp
