import traceback
import torch

from setting import setting
from utils import *
from face_utils.models.inception_resnet_v1 import InceptionResnetV1
from face_utils.models.mtcnn import MTCNN
from face_utils.face_mysql import FaceMysql
from face_utils.face_milvus import FaceMilvus
from face_utils.utils import draw_face_frame, draw_text

root_path = "face_utils/"

config = setting()


# 人脸检测模型
class FaceDetectModel(object):
    def __init__(self):
        # 检测模型参数
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=config.device
        )

    def detect_face(self, orig_image):
        """
        人脸检测
        :param orig_image:
        :return: 人脸的位置与大小信息
        """
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        faces, boxes = self.mtcnn(image)
        return faces, boxes


# 特征提取模型
class FeatureExtractionModel(object):
    def __init__(self):
        self.name = "face recognition model"
        # 人脸特征提取模型参数
        self.device = config.device
        self.recognition_pretrained = config.face_recognition_pretrained
        self.recognition_threshold = config.face_recognition_threshold

        # 初始化模型
        self._load_reconition_model()

    def _load_reconition_model(self):
        # 加载人脸特征提取模型
        if self.recognition_pretrained == "vggface2":
            model_path = root_path + "models/pretrained/20180402-114759-vggface2.pt"
            self.recognition_net = InceptionResnetV1(tmp_classes=8631, device=self.device)
            self.recognition_predictor = self.recognition_net.eval()
        else:
            raise ValueError("The pretrained model is wrong")
        self.recognition_net.load(model_path)

    def extrect_feature(self, faces):
        faces_tensor = torch.stack(faces).to(self.device)
        # print(faces_tensor.shape, type(faces_tensor))
        return np.array(self.recognition_predictor(faces_tensor.float()).detach().cpu())


class ImageFaceRecognition(object):
    def __init__(self):

        self.recognition_threshold = config.face_recognition_threshold
        self.detect_model = FaceDetectModel()
        self.extrect_model = FeatureExtractionModel()

        self.db = FaceMysql(config.db_info, config.face_table_name)
        self.milvus = FaceMilvus(config.milvus_host,
                                 config.milvus_port,
                                 config.milvus_face_param)

    def insert_face(self, image, uid, is_replace):
        """
        录入人脸
        :param is_replace:是否替换已有人脸
        :param image: 输入特征
        :param uid: 人脸id（或人脸标识）
        :return: 录入结果（是否成功）

        """
        try:
            if len(image) == 0:
                return False, 'image is empty', 500, None
            faces, boxes = self.detect_model.detect_face(image)
            face_num = len(faces)
            if faces is None:
                return False, 'detect face is failed', 500, None
            # 特征提取
            face_features = self.extrect_model.extrect_feature(faces)
            if not face_num:
                return False, "There isn't face to be detected in the image", 500, None

            feature_mids, feature_uids, face_num, msg = self.db.load_face_info()  # 读取人脸库信息
            # 是否替换
            if is_replace:
                if face_num > 0:
                    if uid not in list(feature_uids.iloc):
                        msg = "There isn't %d 's face information in the database" % uid
                        return False, msg, 500, None
                    else:
                        face_mid = feature_mids[feature_uids == uid]
                        mids = [int(i) for i in face_mid.iloc]
                        self.face_delete(mids, [uid])
                else:
                    msg = "There isn't face information in the database"
                    return False, msg, 500, None
            if (not is_replace) and uid in list(feature_uids.iloc):
                msg = "There is %d 's face information in the database" % uid
                return False, msg, 500, None

            mil_ret = self.milvus.insert_vectors(face_features)
            if mil_ret == -1:
                return False, "An error occurred while inserting data into thes milvu", 500, None
            else:
                mid = mil_ret[0]
            lastid = self.db.insert_face_info(mid, uid)

            if lastid is None or lastid < 0:
                self.milvus.drop_vectors([mid])
                return False, "An error occurred while inserting data into the database", 500, None
            else:
                return True, 'Success!', 200, None
        except Exception as e:
            print(traceback.format_exc())
            return False, 'have error in the code', 500, None

    def reco_face(self, image):
        # 人脸识别
        uids, markImage = [], None
        try:
            faces, boxes = self.detect_model.detect_face(image)
            if len(faces) == 0:  # 检测不到人脸
                msg = "There is no face to be detected in the image"
                return 200, True, msg, uids, markImage
            # 获取人脸特征
            features_images = self.extrect_model.extrect_feature(faces)
            # 获取数据库的人脸数据
            mids, distances = self.milvus.search_vectors(features_images)
            if isinstance(mids, int):
                if mids == -1:
                    return 500, False, distances, uids, markImage

            feature_mids, feature_uids, face_num, msg = self.db.load_face_info()  # 读取人脸库信息

            # 当数据库里有数据时，进行检测
            if face_num > 0:
                for face, mid, distance, boxe in zip(faces, mids, distances, boxes):
                    if distance < self.recognition_threshold:  # 识别成功
                        face_uids = feature_uids[feature_mids == mid]
                        if len(face_uids) <= 0:
                            msg = "There isn't this mid(%d) in the database" % mid
                            return 500, False, msg, uids, markImage
                        uids.append(str(face_uids.iloc[0]))
                        draw_text(image, boxe, str(face_uids.iloc[0]))

                markImage = draw_face_frame(image, boxes)
                markImage = np_to_base64(cv2.cvtColor(markImage, cv2.COLOR_BGR2RGB))
                return 200, True, None, uids, markImage

            else:
                msg = "There is not face information in the database"
                return 500, False, msg, uids, markImage

        except Exception as e:
            return 500, False, 'have a error during search face', uids, markImage

    def face_delete(self, mids, uids):
        try:
            self.db.delete_info(uids)
            self.milvus.drop_vectors(mids)
            return True, None, 200
        except:
            print(traceback.format_exc())
            return True, 'error in running code', 500
