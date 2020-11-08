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
import re
from operator import itemgetter

import pandas as pd
import pickle
import numpy as np
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.utils
from tqdm.auto import tqdm
from utils import *
from datetime import datetime


class DataGenerator:
    ##def __init__(self, company_code='ITUB4', data_path='./stock_history', output_path='./outputs', strategy_type='original',
    def __init__(self, company_code, data_path='./stock_history', output_path='./outputs', strategy_type='original',
                 update=False):
        self.company_code = company_code
        self.strategy_type = strategy_type
        self.data_path = data_path
        self.BASE_URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED" \
                        "&outputsize=full&apikey=KD1I9P1L06Y003R9&datatype=csv&symbol="  # api key from alpha vantage service
        self.output_path = output_path
        self.start_col = 'open'
        self.end_col = 'eom_21'
        self.update = update
        self.download_stock_data()

        ##calcula os indicadores e os labels
        self.df = self.create_features()
        ##define pixels das imagens
        self.feat_idx = self.feature_selection()
        self.create_images_per_day()
        self.one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
        self.one_hot_enc.fit(self.df['labels'].values.reshape(-1, 1))
        self.batch_start_date = self.df.head(1).iloc[0]["timestamp"]
        self.test_duration_years = 1


    def log(self, text):
            print(text)

    def download_stock_data(self):
        path_to_company_data = self.data_path
        print("caminho para dados da ação:", path_to_company_data)
        parent_path = os.sep.join(path_to_company_data.split(os.sep)[:-1])
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
            print("Diretorio da ação criado", parent_path)

        if not os.path.exists(path_to_company_data):
            self.log("Downloading " + self.company_code)
            download_save(self.BASE_URL + self.company_code, path_to_company_data)
        else:
            self.log("Dados de " + self.company_code + " prontos")

    def calculate_technical_indicators(self, df, col_name, intervals):
        # get_RSI(df, col_name, intervals)  # faster but non-smoothed RSI
        get_RSI_smooth(df, col_name, intervals)  # momentum
        get_williamR(df, col_name, intervals)  # momentum
        get_mfi(df, intervals)  # momentum
        # get_MACD(df, col_name, intervals)  # momentum, ready to use +3
        # get_PPO(df, col_name, intervals)  # momentum, ready to use +1
        get_ROC(df, col_name, intervals)  # momentum
        get_CMF(df, col_name, intervals)  # momentum, volume EMA
        get_CMO(df, col_name, intervals)  # momentum
        get_SMA(df, col_name, intervals)
        ##get_SMA(df, 'open', intervals)
        get_EMA(df, col_name, intervals)
        get_WMA(df, col_name, intervals)
        get_HMA(df, col_name, intervals)
        ##get_TRIX(df, col_name, intervals)  # trend
        get_CCI(df, col_name, intervals)  # trend
        get_DPO(df, col_name, intervals)  # Trend oscillator
        get_kst(df, col_name, intervals)  # Trend
        get_DMI(df, col_name, intervals)  # trend
        ##get_BB_MAV(df, col_name, intervals)  # volatility
        # get_PSI(df, col_name, intervals)  # can't find formula
        ##get_force_index(df, intervals)  # volume
        ##get_kdjk_rsv(df, intervals)  # ready to use, +2*len(intervals), 2 rows
        get_EOM(df, col_name, intervals)  # volume momentum
        get_volume_delta(df)  # volume +1
        ##get_IBR(df)  # ready to use +1

    def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        self.log("creating label with original paper strategy")
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)


        while row_counter < total_rows:

            if row_counter >= window_size - 1:

                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) // 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1

                for i in range(window_begin, window_end + 1):

                    price = df.iloc[i][col_name]

                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[window_middle] = 0
                elif min_index == window_middle:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2


            row_counter = row_counter + 1

            pbar.update(1)

        pbar.close()
        return labels

    def create_labels_price_rise(self, df, col_name):
        """
        labels data based on price rise on next day
          next_day - prev_day
        ((s - s.shift()) > 0).astype(np.int)
        """

        df["labels"] = ((df[col_name] - df[col_name].shift()) > 0).astype(np.int)
        df = df[1:]
        df.reset_index(drop=True, inplace=True)

    def create_label_mean_reversion(self, df, col_name):
        """
        strategy as described at "https://decodingmarkets.com/mean-reversion-trading-strategy"

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        self.log("creating labels with mean mean-reversion-trading-strategy")
        get_RSI_smooth(df, col_name, [3])  # new column 'rsi_3' added to df
        rsi_3_series = df['rsi_3']
        ibr = get_IBR(df)
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        count = 0
        for i, rsi_3 in enumerate(rsi_3_series):
            if rsi_3 < 15:  # buy
                count = count + 1

                if 3 <= count < 8 and ibr.iloc[i] < 0.2:  # TODO implement upto 5 BUYS
                    labels[i] = 1

                if count >= 8:
                    count == 0
            elif ibr.iloc[i] > 0.7:  # sell
                labels[i] = 0
            else:
                labels[i] = 2

        return labels

    def create_label_short_long_ma_crossover(self, df, col_name, short, long):
        """
        if short = 30 and long = 90,
        Buy when 30 day MA < 90 day MA
        Sell when 30 day MA > 90 day MA

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        self.log("creating label with {}_{}_ma".format(short, long))

        def detect_crossover(diff_prev, diff):
            if diff_prev >= 0 > diff:
                # buy
                return 1
            elif diff_prev <= 0 < diff:
                return 0
            else:
                return 2

        get_SMA(df, 'close', [short, long])
        labels = np.zeros((len(df)))
        labels[:] = np.nan
        diff = df['close_sma_' + str(short)] - df['close_sma_' + str(long)]
        diff_prev = diff.shift()
        df['diff_prev'] = diff_prev
        df['diff'] = diff

        res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
        print("labels count", np.unique(res, return_counts=True))
        df.drop(columns=['diff_prev', 'diff'], inplace=True)
        return res

    def create_features(self):
        if not os.path.exists(os.path.join(self.output_path, "df_" + self.company_code+".csv")) or self.update:
            df = pd.read_csv(self.data_path, engine='python')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            intervals = range(6, 22)  # 21
            self.calculate_technical_indicators(df, 'close', intervals)
            self.log("Salvando dataframe...")
            df.to_csv(os.path.join(self.output_path, "df_" + self.company_code+".csv"), index=False)
        else:
            self.log("Indicadores ja calculados. Carregando...")
            df = pd.read_csv(os.path.join(self.output_path, "df_" + self.company_code+".csv"))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

        prev_len = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if 'labels' not in df.columns or self.update:
            if re.match(r"\d+_\d+_ma", self.strategy_type):
                short = self.strategy_type.split('_')[0]
                long = self.strategy_type.split('_')[1]
                df['labels'] = self.create_label_short_long_ma_crossover(df, 'close', short, long)
            else:
                df['labels'] = self.create_labels(df, 'close')

            prev_len = len(df)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            df.drop(columns=['dividend_amount', 'split_coefficient'], inplace=True)
            df.to_csv(os.path.join(self.output_path, "df_" + self.company_code + ".csv"), index=False)
        else:
            print("labels já calculados")

        # pickle.dump(df, open(os.path.join(self.output_path, "df_" + self.company_code), 'wb'))
        # console_pretty_print_df(df.head())
        self.log("Quantidade de indicadores técnicos: {}".format(len(list(df.columns)[7:])))
        return df

    def feature_selection(self):
        ##define o intervalo em anos
        df_batch = self.df_by_date(None, 5)
        list_features = list(df_batch.loc[:, self.start_col:self.end_col].columns)

        ##normaliza os valores entre 0 e 1
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        ##mm_scaler = StandardScaler()  # or StandardScaler?
        x_train = mm_scaler.fit_transform(df_batch.loc[:, self.start_col:self.end_col].values)
        y_train = df_batch['labels'].values
        num_features = 225
        topk = 'all'
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        common = list(set(selected_features_anova).intersection(selected_features_mic))

        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        return feat_idx

    def df_by_date(self, start_date=None, years=5):
        if not start_date:
            start_date = self.df.head(1).iloc[0]["timestamp"]

        end_date = start_date + pd.offsets.DateOffset(years=years)
        df_batch = self.df[(self.df["timestamp"] >= start_date) & (self.df["timestamp"] <= end_date)]
        return df_batch

    def create_images_per_day(self):
        df_batch = self.df_by_date(None, 5)
        x = df_batch.loc[:, 'open':'eom_21'].values
        y = df_batch.loc[:, 'labels'].values

        x = x[:, self.feat_idx]

        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x)
        dim = int(np.sqrt(x.shape[1]))

        x = save_array_as_images(x, dim, dim, './images/' + self.company_code, 'image', y)


    def get_rolling_data_next(self, start_date=None, window_size_yrs=6):
        if not start_date:
            start_date = self.batch_start_date

        df_batch_train = self.df_by_date(start_date, window_size_yrs)
        ##última data de cotação
        train_end_date = df_batch_train.tail(1).iloc[0]["timestamp"]

        ##data_inicio = fim + 1 dia
        test_start_date = train_end_date + pd.offsets.DateOffset(days=1)
        ##data_fim = inicio + 1 ano
        test_end_date = test_start_date + pd.offsets.DateOffset(years=self.test_duration_years)

        is_last_batch = False
        print('tail timestamp:')
        print(self.df.tail(1).iloc[0]["timestamp"])
        print('end date')
        print(test_end_date)
        print((self.df.tail(1).iloc[0]["timestamp"] - test_end_date).days)
        if (self.df.tail(1).iloc[0]["timestamp"] - test_end_date).days < 180:  # 6 months
            is_last_batch = True
        return is_last_batch
