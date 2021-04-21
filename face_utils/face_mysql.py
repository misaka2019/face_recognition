# author:Boyle time:2020/7/31
import pandas as pd
import pymysql
import datetime


# 连接数据库，人脸库持久化
class FaceMysql(object):

    def __init__(self, db_info: dict, table_name="face_recognition"):

        # 如果db_info为None，
        if not db_info:
            db_info = {}

        self.db_info = {}
        self.db_info['host'] = db_info['host'] if 'host' in db_info else "localhost"
        self.db_info['user'] = db_info['user'] if 'user' in db_info else "root"
        self.db_info['password'] = db_info['password'] if 'password' in db_info else "123456"
        self.db_info['port'] = db_info['port'] if 'port' in db_info else 3306
        self.db_info['database'] = db_info['database'] if 'database' in db_info else "face"
        self.db_info['charset'] = db_info['charset'] if 'charset' in db_info else "utf8"
        self.table_name = table_name
        self.init_db()

    def conn_mysql(self):
        conn = pymysql.connect(**self.db_info)
        return conn

    def init_db(self):
        conn = self.conn_mysql()
        sql = ''' 
            CREATE TABLE IF NOT EXISTS `%s` (
                  `id` BIGINT(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
                  `mid` BIGINT(32) DEFAULT NULL COMMENT 'milvus的id',
                  `uid` BIGINT(32) DEFAULT NULL COMMENT '人脸id',
                  `date` DATETIME DEFAULT NULL COMMENT '建立时间',
                  `state` TINYINT(1) DEFAULT NULL COMMENT '人脸状态',
                  PRIMARY KEY (`id`)
                ) ENGINE=INNODB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
            ''' % (self.table_name)
        cursor = conn.cursor()
        try:
            # 执行sql语句
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            # 如果发生错误，则回滚事务
            print(e.args)
            conn.rollback()
        conn.close()

    def insert_face_info(self, mid, uid):
        conn = self.conn_mysql()
        try:
            cursor = conn.cursor()
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql = "INSERT `%s`(`mid`, `uid`, `date`, `state`) " \
                  "VALUES('%d','%d','%s','%d');" % (self.table_name,
                                                    mid, uid, date, 1)
            # 执行sql语句
            cursor.execute(sql)
            lastid = int(cursor.lastrowid)

            conn.commit()
        except Exception as e:
            # 如果发生错误，则回滚事务
            print(e.args)
            lastid = -1
            conn.rollback()
        conn.close()
        return lastid

    def load_face_info(self):
        db = self.conn_mysql()
        # 获得数据库的人脸数据
        try:
            # cursor.execute(sql)
            # results = cursor.fetchall()
            sql = "select * from `%s` where state=1" % self.table_name
            face_info = pd.read_sql(sql, db)
        except:
            msg = "An error occurred while search data from the database"
            db.close()
            return None, None, -1, msg
        db.close()
        # 获得数据库的特定字段内容
        try:
            mids = face_info['mid']
            uids = face_info['uid']
            face_num = face_info.shape[0]
        except NameError:
            msg = "field error in database"
            return None, None, -1, msg
        return mids, uids, face_num, None

    def delete_info(self, uids):
        conn = self.conn_mysql()
        try:
            cursor = conn.cursor()
            for uid in uids:
                sql = "DELETE FROM `%s` where uid=%d" % (self.table_name, uid)
                cursor.execute(sql)
                conn.commit()
        except:
            msg = "An error occurred while delete data from the database"
            return None, None, -1, msg
