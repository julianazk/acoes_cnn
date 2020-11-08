#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import compute_class_weight

from data_generator import DataGenerator, plt
##from logger import Logger
##from secrets import api_key
from utils import download_save, seconds_to_minutes

##configuração do tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

##recebe os argumentos da chamada da função train.py "nome da ação" "estratégia"
##exemplo: python train.py RENT3.SAO original, onde RENT3.SAO é o nome da ação no alpha advantage (código bovespa + .SAO)
args = sys.argv  # a list of the arguments provided (str)
print("running stock_cnn.py", args)
pd.options.display.width = 0
company_code = args[1]
strategy_type = args[2]
ROOT_PATH = ".."
iter_changes = "fresh_rolling_train"  # label for changes in this run iteration
INPUT_PATH = os.path.join(ROOT_PATH, "stock_history", company_code)
OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs", iter_changes)
LOG_PATH = OUTPUT_PATH + os.sep + "logs"
LOG_FILE_NAME_PREFIX = "log_{}_{}_{}".format(company_code, strategy_type, iter_changes)
PATH_TO_STOCK_HISTORY_DATA = os.path.join(ROOT_PATH, "stock_history")

##se caminho de input não existe, cria
if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
    print("Input Directory created", INPUT_PATH)

##se caminho de output não existe, cria
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Output Directory created", OUTPUT_PATH)

##URL do alphaadvantage para obter valores
##chave fixa KD1I9P1L06Y003R9
BASE_URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED" \
           "&outputsize=full&apikey=KD1I9P1L06Y003R9&datatype=csv&symbol="  # api key from alpha vantage service
data_file_name = company_code + ".csv"
PATH_TO_COMPANY_DATA = os.path.join(PATH_TO_STOCK_HISTORY_DATA, company_code, data_file_name)
print(INPUT_PATH)
print(OUTPUT_PATH)
print(PATH_TO_COMPANY_DATA)


start_time = time.time()
##chama script datagenerator
data_gen = DataGenerator(company_code, PATH_TO_COMPANY_DATA, OUTPUT_PATH, strategy_type, False)





