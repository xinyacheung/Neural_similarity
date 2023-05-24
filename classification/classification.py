from sklearn.linear_model import LogisticRegression
import numpy as np
import glob
import re
import time as ti
import pickle
import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from xgboost.sklearn import XGBClassifier
import os
dir = os.path.dirname(__file__)

start_time=ti.time()

save_dir = dir + '/data'

bin_size = 0.05

num_feature = 30
color = ['tab:blue','tab:red','tab:orange']

# '''
for roi in ['PFC','FEF','LIP', 'MT', 'IT', 'V4']:#:#, 'LIP', 'MT', 'IT', 'V4']:
    # train classifier
    for task in ['motion', 'color']:#['task']
        time_series = np.load(save_dir +f'/{task}_time_series_{roi}_{bin_size}.npy')
        label = np.load(save_dir +f'/{task}_label_{roi}_{bin_size}.npy')
        print(f'cost time:{ti.time() - start_time}s')
        print(time_series.shape)
        print(label)

        transformer = Normalizer().fit(time_series)
        time_series = transformer.transform(time_series)

        if task =='motion':
            clf_m = LogisticRegression().fit(time_series, label)
            print(f'motion clf:{clf_m.score(time_series, label)}')
        if task =='color':
            clf_c = DecisionTreeClassifier().fit(time_series, label)
            print(f'color clf:{clf_c.score(time_series, label)}')
        if task =='task':
            clf_t = DecisionTreeClassifier().fit(time_series, label)
            print(f'task clf:{clf_t.score(time_series, label)}')