from sklearn.cluster import KMeans
import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd

class Cluster:
    try:
        def __init__(self):
            pass

        def make_cluster(self,train_arr,test_arr,n_cluster=3):
            kmean=KMeans(n_clusters=n_cluster,random_state=42)
            cluster_labels_train=kmean.fit_predict(train_arr)
            cluster_labels_test=kmean.predict(test_arr)
            logging.info("cluster labels are created")

            train_arr_with_cluster=np.hstack([train_arr,cluster_labels_train.reshape(-1,1)])
            test_arr_with_cluster=np.hstack([test_arr,cluster_labels_test.reshape(-1,1)])


            return (train_arr_with_cluster,test_arr_with_cluster)
    
    except Exception as e:
        raise CustomException(e,sys)



        