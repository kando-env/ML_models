import json
import time
from datetime import date, datetime, time, timedelta, timezone
from kando_data.model_runner.cloud import CloudEnv

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle5 as pickle
import sys

sys.path.insert(0, '..')
# from kando_data.model_runner.local import ModelTemplate
from model_template import ModelTemplate
from scipy import interpolate
import os
from pathlib import Path
from dotenv import load_dotenv


class CodRegressorUsingBh(ModelTemplate):
    def __init__(self):
        super().__init__()
        self.model_params_path = '/api/training/ml_models/model_params'
        self.model = self._load_obj('cod_model')
        self.pred = None
        self.sites_list = (
            'sorek', 'sorek_new', 'beitGalla_dom', 'beitGalla', 'refaim_oil', 'refaim_150', 'begin', 'atarot',
            'vetrinari')

        self.target = 'COD'
        self.num_samples_week = 4 * 24 * 7
        self.filename_model_chosen_features = 'lgb_100_chosen_params_13-01-2021'

    def do_train(self, **kwargs):
        """
        trains the model on entire dataset from sites listed in sites_list, not including last 2 weeks (or 0.2% of data)
        :param kwargs: no inputs currently required
        """
        self.model_params_path = '/api/training/ml_models/model_params'
        x_train_all, y_train_all, x_eval_all, y_eval_all, x_test_all, y_test_all = self._create_training_df()
        training_features = self._load_obj(self.filename_model_chosen_features)
        lightgbm_params = {'n_estimators': 17320, 'max_depth': 9, 'colsample_bytree': 0.777, 'learning_rate': 0.007,
                           'gamma': 0, 'min_child_weight': 0, 'subsample': 0.777, "early_stopping_rounds": 673,
                           "metric": 'rmse'}
        self.model = lgb.LGBMRegressor(**lightgbm_params)
        x_train = x_train_all[training_features]
        x_eval = x_eval_all[training_features]
        x_test = x_test_all[training_features]
        self.model.fit(x_train, y_train_all, eval_set=[(x_train, y_train_all), (x_test, y_test_all),
                                                       (x_eval, y_eval_all)], verbose=300)
        self._save_obj(self.model, 'cod_model')

    def do_predict(self, kwargs):
        """

        :param kwargs: required site, start and end timestamps for prediction.
        site should currently be one of the options in sites_list
        :return: json with prediction outputs
        """

        self.model_params_path = '/training/ml_models/model_params'
        df = self._get_site_df(kwargs['site'], test_start=kwargs['start'], test_end=kwargs['end'])
        df, base_features = self._extract_rolling_features(df)
        if self.target in df:
            x_test = df.drop(columns=[self.target], axis=1)
        else:
            x_test = df.copy()
        x_test = self._transform_dataframe(x_test, base_features)
        training_features = self._load_obj(self.filename_model_chosen_features)

        x_test = x_test[training_features]
        prediction = self.model.predict(x_test)
        prediction = pd.DataFrame({self.target: prediction}, index=x_test.index)
        return prediction.to_json()

    # @staticmethod
    def _extract_rolling_features(self, df):
        """
        extracts rolling features such as mean and std from a certain time window
        :param df: dataframe containing target and basic features (base_features) such as 'BH'
        :return: dataframe with additional rolling features, and base_features
        """

        base_features = set(df.columns) - {'COD'}
        for window_size in [672, 112, 12, 32, 80, 3, 48, 16, 4, 84, 96, 72]:
            for col in base_features:
                rolling_window = df[col].rolling(window=window_size, min_periods=1)
                abs_diff_rolling_window = df[col].diff().abs().rolling(window=window_size, min_periods=1)
                diff_rolling_window = df[col].diff().rolling(window=window_size, min_periods=1)

                df[f'{col}_window{window_size}_mean'] = rolling_window.mean()
                df[f'{col}_window{window_size}_std'] = rolling_window.std()
                df[f'{col}_window{window_size}_median'] = rolling_window.median()
                df[f'{col}_window{window_size}_max'] = rolling_window.max()
                df[f'{col}_window{window_size}_min'] = rolling_window.min()
                df[f'{col}_window{window_size}_sum'] = rolling_window.sum()

                df[f'diff2_{col}_window{window_size}_mean'] = df[f'{col}'] - rolling_window.mean()
                df[f'diff2_{col}_window{window_size}_median'] = df[f'{col}'] - rolling_window.median()

                df[f'diff_{col}_window{window_size}_abssum'] = abs_diff_rolling_window.sum() / np.sqrt(window_size)
                df[f'diff_{col}_window{window_size}_mean'] = diff_rolling_window.mean()
                df[f'diff_{col}_window{window_size}_median'] = diff_rolling_window.median()

                df[f'{col}_prop2_window{window_size}_mean'] = df[f'{col}'] / df[f'{col}'].shift(1).rolling(
                    window=window_size, min_periods=1).mean()
                df[f'{col}_prop2_window{window_size}_median'] = df[f'{col}'] / df[f'{col}'].shift(1).rolling(
                    window=window_size, min_periods=1).median()
                df[f'{col}_prop2_window{window_size}_max'] = df[f'{col}'] / df[f'{col}'].shift(1).rolling(
                    window=window_size, min_periods=1).max()
                df[f'{col}_prop2_window{window_size}_min'] = df[f'{col}'] / df[f'{col}'].shift(1).rolling(
                    window=window_size, min_periods=1).min()
        return df, base_features

    def _save_obj(self,obj, name):
        """
        saves and obj to model_params folder
        """
        p = Path(name + '.pkl').resolve()
        print("full path to be saved:" +str(p))

        with open(os.getcwd() + self.model_params_path + '/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self,name):
        """
        loads an obj to model_params folder
        """
        p = Path(name + '.pkl').resolve()
        print("full path to be loaded:" +str(p))
        with open(os.getcwd() + self.model_params_path + '/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def _create_training_df(self):
        """
        creates df from all sites in sites_lists, required for training
        """
        test_size = 0.1
        validation_size = 0.1
        x_train_all = pd.DataFrame()
        x_eval_all = pd.DataFrame()
        x_test_all = pd.DataFrame()
        y_train_all = pd.Series(dtype=float, name=self.target)
        y_eval_all = pd.Series(dtype=float, name=self.target)
        y_test_all = pd.Series(dtype=float, name=self.target)

        for site in self.sites_list:
            df = self._get_site_df(site=site)

            # choose a method out of ['diff','peak_detection','rolling_std']
            # dict where key is you path and value is dataframe
            df, base_features = self._extract_rolling_features(df)
            x_train, y_train, x_eval, y_eval, x_test, y_test = self._test_train_time_split(df, test_size,
                                                                                           validation_size)
            x_train = self._transform_dataframe(x_train, base_features)
            x_eval = self._transform_dataframe(x_eval, base_features)
            x_test = self._transform_dataframe(x_test, base_features)

            x_train, y_train, x_eval, y_eval, x_test, y_test = self._df_dropna(x_train, y_train, x_eval, y_eval, x_test,
                                                                               y_test)

            x_train_all = pd.concat([x_train_all, x_train], ignore_index=True)
            y_train_all = pd.concat([y_train_all, y_train], ignore_index=True)
            x_eval_all = pd.concat([x_eval_all, x_eval], ignore_index=True)
            y_eval_all = pd.concat([y_eval_all, y_eval], ignore_index=True)
            x_test_all = pd.concat([x_test_all, x_test], ignore_index=True)
            y_test_all = pd.concat([y_test_all, y_test], ignore_index=True)

        return x_train_all, y_train_all, x_eval_all, y_eval_all, x_test_all, y_test_all

    def _df_dropna(self, x_train, y_train, x_eval, y_eval, x_test, y_test):
        """
        """
        train_df = x_train.copy()
        train_df[self.target] = y_train
        train_df.dropna(subset=[self.target], inplace=True)
        X_train = train_df.drop([self.target], axis=1)
        y_train = train_df[self.target]

        eval_df = x_eval.copy()
        eval_df[self.target] = y_eval
        eval_df.dropna(subset=[self.target], inplace=True)
        X_eval = eval_df.drop([self.target], axis=1)
        y_eval = eval_df[self.target]

        test_df = x_test.copy()
        test_df[self.target] = y_test
        test_df.dropna(subset=[self.target], inplace=True)
        X_test = test_df.drop([self.target], axis=1)
        y_test = test_df[self.target]

        print(f'X_train shape: {X_train.shape},X_eval: {X_eval.shape},X_test: {X_test.shape}')
        print(f'y_train shape: {y_train.shape},y_eval: {y_eval.shape},y_test: {y_test.shape}')

        return X_train, y_train, X_eval, y_eval, X_test, y_test

    def _test_train_time_split(self, df, test_size, validation_size):
        """
        splits dataframe to train, validation and test sets
        """
        if len(df) > (4 * self.num_samples_week):
            train_index = list(range(0, len(df) - 2 * self.num_samples_week))
            validation_index = list(range(len(df) - 2 * self.num_samples_week, len(df) - self.num_samples_week))
            test_index = list(range(len(df) - self.num_samples_week, len(df)))
        else:

            train_index = list(range(0, int(len(df) * (1 - (test_size + validation_size)))))
            validation_index = list(range(int(len(df) * (1 - (test_size + validation_size))),
                                          int(len(df) * (1 - test_size))))
            test_index = list(range(int(len(df) * (1 - test_size)), len(df) - 1))

        train_set = df.iloc[train_index].copy()
        eval_set = df.iloc[validation_index].copy()
        test_set = df.iloc[test_index].copy()

        x_train = train_set.drop(columns=[self.target], axis=1)
        y_train = train_set[[self.target]]
        x_eval = eval_set.drop(columns=[self.target], axis=1)
        y_eval = eval_set[[self.target]]
        x_test = test_set.drop(columns=[self.target], axis=1)
        y_test = test_set[[self.target]]

        print(f'x_train shape: {x_train.shape},x_eval shape: {x_eval.shape},x_test shape: {x_test.shape}')
        print(f'y_train shape: {y_train.shape},y_eval shape: {y_eval.shape},y_test shape: {y_test.shape}')

        return x_train, y_train, x_eval, y_eval, x_test, y_test

    def get_meta_data(self):
        pass

    def _get_site_df(self, site, test_start=None, test_end=None):
        """
        gets bh, cod and wl metadata and extracts basic features to be used for training or validation
        """
        print("Path at terminal when executing this file")
        print(os.getcwd() + "\n")
        p = Path('dict_bh_info.json').resolve()
        print("full path to be loaded:" +str(p))
        with open(os.getcwd() + self.model_params_path + '/dict_bh_info.json', 'r') as fp:
            dict_bh_info = json.load(fp)

        if test_start is None:
            # this means we are predicting
            start = dict_bh_info[site]['start']
            end = dict_bh_info[site]['end']
        else:
            # this means we are training
            start = test_start
            end = test_end



        print(f'creating df for site {site}')
        bh_data = self.client.get_all(point_id=dict_bh_info[site]['bh_pointid'], unit_id='', start=start, end=end,
                                      raw_data='True')
        df_bh = pd.DataFrame(bh_data['samplings']).T

        initial_index, last_index = df_bh.BH_FDOM_SIG.first_valid_index(), df_bh.BH_FDOM_SIG.last_valid_index()
        print(f'requested BH start time: {datetime.fromtimestamp(int(start), timezone.utc)}')
        print(f'BH initial index: {datetime.fromtimestamp(int(df_bh.index[0]), timezone.utc)}')

        df_bh = df_bh.loc[initial_index:last_index]
        df_bh.loc[:, 'BH'] = df_bh['BH_FDOM_SIG']
        df_bh.drop(['BH_FDOM_SIG'], axis=1)

        if 'WL' in df_bh:
            df_bh = df_bh[['BH', 'BH_FDOM_BKG', 'WL', 'DateTime']]
        else:
            df_bh = df_bh[['BH', 'BH_FDOM_BKG', 'DateTime']]

        df_bh = self._rearrange_time_index(df_bh)

        if site == 'refaim' or site == 'refaim_150':
            # move refaim half an hour to match the COD station
            df_bh = df_bh.shift(periods=2)

        start = pd.Timestamp(df_bh.index[0], tz=timezone.utc).to_pydatetime().timestamp()
        print(f'start time index requested from controller: {df_bh.index[0]}')

        controller_data = self.client.get_all(point_id=dict_bh_info[site]['cod_pointid'], unit_id='', start=start,
                                              end=end,
                                              raw_data='True')
        df_controller = pd.DataFrame(controller_data['samplings']).T
        print(
            f'controller initial index: {df_controller.index[0]}, DateTime: '
            f'{datetime.fromtimestamp(int(df_controller.index[0]), timezone.utc)}')

        df_controller = df_controller[['EC', 'PH', 'COD', 'TSS', 'TEMPERATURE', 'DateTime', 'ORP', 'PI']]

        df_controller = self._rearrange_time_index(df_controller)
        df = df_controller.join(df_bh)
        print(f'full df initial index: {df.index[0]}')
        print(f'BH first valid index: {df.BH.first_valid_index()}')
        if site == 'sorek' or site == 'refaim' or site == 'refaim_oil' or site == 'refaim_150' or site == 'sorek_new':
            rof_data = self.client.get_all(point_id=dict_bh_info[site]['rof_pointid'], unit_id='', start=start, end=end,
                                           raw_data='True')
            df_rof = pd.DataFrame(rof_data['samplings']).T
            df_rof = df_rof[['WL', 'DateTime']].copy()
            df_rof = self._rearrange_time_index(df_rof)
            # df_rof.index = pd.to_datetime(df_rof.index, unit='s')
            # df_rof = df_rof.resample('15T').nearest()
            if 'WL' in df:
                df = df.drop(['WL'], axis=1)
            df = df.join(df_rof)

        df['DateTime'] = [pd.Timestamp(index, tz=timezone.utc).to_pydatetime().timestamp()
                          for index in df.index]
        if dict_bh_info[site]:
            df = self._remove_corrupt_data(df, site)
        df = self._remove_outliers(df)
        initial_index, last_index = df.BH.first_valid_index(), df.BH.last_valid_index()
        df = df.loc[initial_index:last_index]
        df = df.replace([np.inf, -np.inf], np.nan)

        if site == 'begin':
            # we don't have wl in begin yet

            baseline_wl = 20
            wl_installation_datetime = pd.to_datetime('2021-01-09 22:00:00').timestamp()
            df['WL'][df.DateTime < wl_installation_datetime] = baseline_wl
        df['WL'] = df['WL'].fillna(df['WL'].median(skipna=True))

        df = df[['BH', 'COD', 'EC', 'PH', 'BH_FDOM_BKG', 'WL', 'DateTime']].copy()

        df = self._scale_bh(df, site)
        df = df[['BH', 'COD', 'BH_FDOM_BKG', 'WL']].copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df['logBH'] = np.log1p(df['BH'])
        df['zscore_logBH'] = (df['logBH'] - df['logBH'].std(skipna=True)) / df['logBH'].std(skipna=True)
        for col in ['BH', 'BH_FDOM_BKG', 'WL', 'logBH', 'zscore_logBH']:
            df[f'diff_{col}'] = df[col].diff()

        return df

    @staticmethod
    def _rearrange_time_index(df, keep_date_time=False):
        """
        arranges the timestamp indexes in df to be consistently every 15 minutes (900 seconds)
        """
        # arrange time index to be consistent every dt, for e.g. 15 minutes
        diff_in_seconds = 900

        start_ind = int(df.index[np.where(df.DateTime.diff() == diff_in_seconds)[0][0]])
        end_ind = int(df.index[-1])
        skeleton = pd.DataFrame(index=range(start_ind, end_ind, diff_in_seconds))

        df['DateTime_ds'] = df['DateTime'].astype(int)
        df = df.merge(skeleton, left_on='DateTime_ds', right_index=True, how='outer')
        df.index = df.DateTime_ds
        if keep_date_time:
            df = df.drop(['DateTime_ds'], axis=1)
        else:
            df = df.drop(['DateTime_ds', 'DateTime'], axis=1)

        df.index = pd.to_datetime(df.index, unit='s')
        df = df.resample('15T').nearest()
        return df

    def _remove_corrupt_data(self,df, site):
        """
        removes corrupted data as specified in dict_bh_info per each site
        """
        df_sliced = df.copy()
        df_excluded = pd.DataFrame()
        p = Path('dict_bh_info.json').resolve()
        print("full path to be loaded:" +str(p))

        with open(os.getcwd() + self.model_params_path+'/dict_bh_info.json', 'r') as fp:
            dict_bh_info = json.load(fp)

        for time_str in dict_bh_info[site]['corrupt_data_ind']:
            if len(time_str) < 1:
                break
            startdate_time_str, end_date_time_str = time_str[0], time_str[1]
            start_date_time = pd.to_datetime(startdate_time_str).timestamp()
            end_date_time = pd.to_datetime(end_date_time_str).timestamp()
            df_excluded = df_excluded.append(
                df_sliced[(df_sliced.DateTime >= start_date_time) & (df_sliced.DateTime <= end_date_time)],
                verify_integrity=True)
            df_sliced.loc[
                (df_sliced.DateTime >= start_date_time) & (df_sliced.DateTime <= end_date_time), ['BH', 'COD']] = None

        return df_sliced

    @staticmethod
    def _remove_outliers(df):
        """
        removes outliers from dataframes, and replaces with None
        :param df:
        :return:
        """
        df.loc[(df.COD == 1) | (df.COD == 10) | (df.COD > 4000), ['COD']] = None
        df.loc[df.BH < 500, ['BH']] = None
        df.loc[df.WL < 0, ['WL']] = None
        return df

    def _scale_bh(self, df, site):
        """
        scales bh reading according to distance per sensor

        """
        scaling_df = self._load_obj('scaling_df')
        with open(os.getcwd() + self.model_params_path+'/dict_bh_info.json', 'r') as fp:
            dict_bh_info = json.load(fp)

        if site == 'atarot':
            height = df.DateTime.apply(lambda curr_time: 150
            if pd.to_datetime('2021-01-05 15:00:00').timestamp() <= curr_time else dict_bh_info[site]['height'])
            dis_from_sensor = height - df['WL']
        else:
            dis_from_sensor = (dict_bh_info[site]['height'] - df['WL'])
        dis_from_sensor[np.isnan(dis_from_sensor)] = np.nanmedian(dis_from_sensor)
        x = scaling_df.index
        y = scaling_df['Average'].values
        f_avg = interpolate.interp1d(x, y, fill_value="extrapolate")
        if dict_bh_info[site]['sensor_num'] is None:
            y_sensor = scaling_df['Average']
        else:
            y_sensor = scaling_df['BH' + str(dict_bh_info[site]['sensor_num'])]
        f_sensor = interpolate.interp1d(x, y_sensor, fill_value="extrapolate")
        intercal = f_sensor(250) / f_sensor(dis_from_sensor)

        first_outcome = df.BH * intercal
        second_outcome = first_outcome * f_avg(250) / f_sensor(250)
        if site == 'refaim_oil':
            df['BH'] = 2.83 * second_outcome
        elif site == 'beitGalla_dom':
            df['BH'] = 2.83 * second_outcome
        else:
            df['BH'] = second_outcome
        return df

    @staticmethod
    def _transform_dataframe(df, base_features):
        """
        retrieves additional features from df such as division of features and lag variables
        :param df:
        :param base_features:
        :return:
        """
        param_pairs = [[a, b] for a in base_features
                       for b in base_features if a != b]
        for param1, param2 in param_pairs:
            df[f'{param1}/{param2}'] = df[param1] / df[param2]

        for col in base_features:
            for j in [1, 2, 3, 4, 5, 11, 85, 83, 69, 96, 79, 76, 78, 16, 82, 6, 36,
                      7, 63, 77, 38, 20, 34, 75, 18, 40, 24, 66, 84, 32, 88, 97, 179, 87]:
                new_name = f'{col}(t-{j})'
                df[new_name] = df[col].shift(j)
                df[f'{col}/{new_name}'] = (df[col] / df[new_name])

        df = df.replace([np.inf, -np.inf], np.nan)

        return df


def get_start_and_end_time():
    end = datetime.combine(date.today(), time.min)
    start = end - timedelta(weeks=16)
    return start.timestamp(), end.timestamp()

def main():
    # dotenv_path = join(dirname(__file__), '.env')
    load_dotenv()
    c_env = CloudEnv()
    #
    MODEL_NAME = "cod_regressor_using_bh"
    #
    # c_env.train_model(MODEL_NAME, params={}, machine_type="c5.4xlarge")
    #
    # c_env.deploy_model("mosra3h0k3knhd9")

    site = 'begin'
    start = datetime(2021, 1, 7, 0, 0).timestamp()
    end = ''
    y_pred = c_env.request_prediction(deployment_id='debuo03nv2z3zz',
                                      context={"model": MODEL_NAME, "site": site, "start": start, "end": end})
    print(y_pred)

if __name__ == "__main__":
    main()