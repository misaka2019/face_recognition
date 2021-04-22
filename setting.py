db_info = {"host": "127.0.0.1", "user": "root", "password": "123456", "port": 3306,
           "database": "face", "charset": "utf8"}

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
        self.milvus_host = "127.0.0.1"
        self.milvus_port = "19530"
        self.milvus_face_collection = "face_info"

        self.device = 'cpu'

        self.face_size = 160
        self.face_recognition_pretrained = "vggface2"
        self.face_recognition_threshold = 1.1
        self.face_table_name = "face_information"
