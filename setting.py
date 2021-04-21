db_info = {"host": "10.21.23.210", "user": "archive-system", "password": "archive-system-backend", "port": 3306,
           "database": "archive-system", "charset": "utf8"}

# 人脸识别
milvus_param = {'collection_name': 'face_info', 'dimension': 512,
                'index_file_size': 2048, 'metric_type': 'L2'}


class setting(object):

    def __init__(self):
        self.host = '0.0.0.0'
        self.app_port = 8000
        # 人脸识别
        self.db_info = db_info
        self.milvus_face_param = milvus_param
        self.milvus_host = "10.21.23.241"
        self.milvus_port = "19530"
        self.milvus_face_collection = "face_info"

        self.face_detect_net_type = "RFB"
        self.device = 'cpu'
        self.face_detect_candidate_size = 1500
        self.face_detect_input_size = 640
        self.face_detect_threshold = 0.8

        self.face_size = 160
        self.face_recognition_pretrained = "vggface2"
        self.face_recognition_threshold = 1.1
        self.face_table_name = "face_information"
