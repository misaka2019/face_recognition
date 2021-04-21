from milvus import Milvus, MetricType


class FaceMilvus(object):
    def __init__(self, host, port, param):
        # 连接milvus
        self.collection = self.param['collection_name']
        self.milvus = Milvus(self.host, self.port)
        self.host = host
        self.port = port
        self.param = param
        if "metric_type" in self.param:
            if self.param["metric_type"] == "L2":
                self.param["metric_type"] = MetricType.L2
            elif self.param["metric_type"] == "Jaccard":
                self.param["metric_type"] = MetricType.JACCARD
            elif self.param["metric_type"] == "Hamming":
                self.param["metric_type"] = MetricType.HAMMING
            else:
                print("the metric type of milvus is error")

    def insert_vectors(self, vectors):
        """
        向 milvus 插入向量
        :param vectors: 512 维向量，numpy数组类型
        :return: 插入成功的milvus id，-1表示插入失败
        """
        status, vec_num = self.milvus.count_entities(self.collection)
        if status.code != 0:
            return -1
        ids = list(range(vec_num + 1, vec_num + 1 + len(vectors)))
        mids = []
        for i, mid in enumerate(ids):
            vector = vectors[i].reshape(1, -1).tolist()
            status, _mids = self.milvus.insert(collection_name=self.collection, records=vector, ids=[mid])
            if len(_mids) == 1:
                mids.append(_mids[0])
            else:
                return -1
        if status.code != 0:
            return -1
        else:
            return mids

    def search_vectors(self, vectors, msg=None):
        """
        在 milvus 中搜索最近向量
        :param msg:
        :param vectors: 待搜索向量组
        :return: 最近向量的id以及距离
        """
        status, vec_num = self.milvus.count_entities(self.collection)
        if vec_num == 0:  # milvus中为空
            msg = 'the face database is empty'
            return -1, msg
        search_param = {'nprobe': 512}
        status, results = self.milvus.search(collection_name=self.collection,
                                             query_records=vectors,
                                             top_k=1,
                                             params=search_param)
        if status.code != 0:
            return -1, msg
        else:
            mids = list(map(lambda x: x[0].id, results))
            distances = list(map(lambda x: x[0].distance, results))
            return mids, distances

    def drop_vectors(self, mids):
        self.milvus.delete_entity_by_id(collection_name=self.collection, id_array=mids)
