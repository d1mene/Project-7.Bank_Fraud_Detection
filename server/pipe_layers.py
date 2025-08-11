import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import ensemble, preprocessing, decomposition

class MissingValuesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.medians_['current_address_months_count'] = df.loc[df['current_address_months_count'] != -1, 
                                                               'current_address_months_count'].median()
        self.medians_['session_length_in_minutes'] = df.loc[df['session_length_in_minutes'] != -1, 
                                                            'session_length_in_minutes'].median()

        return self

    def transform(self, X):
        df = X.copy()
        df['source'] = (df['source'] == 'INTERNET').astype(int)
        df.drop('prev_address_months_count', axis=1, inplace=True)

        df['current_address_months_count'].replace(-1, self.medians_['current_address_months_count'], inplace=True)
        df['session_length_in_minutes'].replace(-1, self.medians_['session_length_in_minutes'], inplace=True)
        df['device_distinct_emails_8w'].replace(-1, 1, inplace=True)

        return df
    
    
    
    
class BankMonthsInputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = ensemble.RandomForestRegressor(n_estimators=40, 
                                                        criterion='friedman_mse', 
                                                        max_depth=8,
                                                        min_samples_split=25000
                                                        )
        self.drop_list = None
        
    def fit(self, X, y=None):
        train_df = X.copy()
        cat_list = [x for x in train_df.columns if train_df[x].dtype == 'object']
        train_df = train_df[train_df['bank_months_count'] != -1]
        self.drop_list = ['bank_months_count', 'device_fraud_count'] + cat_list
        
        X_train, y_train = train_df.drop(self.drop_list, axis=1), train_df['bank_months_count']
        self.model.fit(X_train, y_train)
        return self
    
    def transform(self, X):
        df = X.copy()
        X_test = df[df['bank_months_count'] == -1]
        
        if self.drop_list is None:
            raise ValueError('Must use fit on train data first!')
        
        if not X_test.empty:
            X_test.drop(self.drop_list, axis=1, inplace=True)
            pred = self.model.predict(X_test)
            df.loc[df['bank_months_count'] == -1, 'bank_months_count'] = pred
        
        return df




class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, features_list):
        self.features_list = features_list
        self.income_map = {}
        self.quantile_similarity = None
        self.quantile_bank_months = None
        self.power_transformer = None
        self.pca_velocity = None
        # список признаков, связанных со скоростями, которые нужно разложить на главные компоненты
        self.vel_features = [
            'velocity_6h', 'velocity_24h', 'velocity_4w',
            'velocity_day_change', 'velocity_term_month_change',
            'velocity_day_ratio', 'velocity_term_month_ratio'
        ]
        self.bank_branch_bins = None

    # функция, кодирующие порядковый признак по возрастанию процента мошеннических атак
    def _get_fraud_ranking(self, X, y, feature):
        df = X.copy()
        
        df['fraud_bool'] = y
        fraud_percentage_month = df.groupby(feature)['fraud_bool'].mean()
        # сортируем и ранжируем по процентам фрод по месяцам
        fraud_percentage_month = fraud_percentage_month.sort_values().rank()
        # заполняем пропуски
        fraud_percentage_month = fraud_percentage_month.fillna(0).astype('uint8') - 1
        
        return fraud_percentage_month.to_dict()

        
    def fit(self, X, y):
        df = X.copy()

        df['velocity_day_change'] = df['velocity_6h'] - df['velocity_24h']
        df['velocity_term_month_change'] = df['velocity_6h'] - df['velocity_4w']
        df['velocity_day_ratio'] = df['velocity_6h']/df['velocity_24h']
        df['velocity_term_month_ratio'] = df['velocity_6h']/df['velocity_4w']
        
        self.income_map = self._get_fraud_ranking(X, y, 'income')
        
        self.quantile_similarity = preprocessing.QuantileTransformer(output_distribution='normal')
        self.quantile_similarity.fit(df[['name_email_similarity']])
        
        self.quantile_bank_months = preprocessing.QuantileTransformer(output_distribution='normal')
        self.quantile_bank_months.fit(df[['bank_months_count']])
        
        self.power_transformer = preprocessing.PowerTransformer(method='yeo-johnson')
        self.power_transformer.fit(df[['current_address_months_count']])

        self.pca_velocity = decomposition.PCA(n_components=4, svd_solver='arpack')
        self.pca_velocity.fit(df[self.vel_features])
        
        # считаем процентили, чтобы гарантированно получить 4 бина
        bins = np.percentile(df['bank_branch_count_8w'], [0, 25, 50, 75, 100])
        # убираем повторяющиеся границы (если они есть)
        self.bank_branch_bins = np.unique(bins)
        
        return self


    def transform(self, X):
        df = X.copy()
        # создаем фичи, связанные со скоростями
        df['velocity_day_change'] = df['velocity_6h'] - df['velocity_24h']
        df['velocity_term_month_change'] = df['velocity_6h'] - df['velocity_4w']
        df['velocity_day_ratio'] = df['velocity_6h']/df['velocity_24h']
        df['velocity_term_month_ratio'] = df['velocity_6h']/df['velocity_4w']
        
        # бинарные признаки-флаги
        df['only_one_valid'] = df['phone_home_valid'] ^ df['phone_mobile_valid']
        
        df['housing_status_BA'] = (df['housing_status'] == 'BA').astype('uint8')
        df['housing_status_BE'] = (df['housing_status'] == 'BE').astype('uint8')
        df['payment_type_AA'] = (df['payment_type'] == 'AA').astype('uint8')
        df['device_os_windows'] = (df['device_os'] == 'windows').astype('uint8')
        df['device_os_linux'] = (df['device_os'] == 'linux').astype('uint8')
        df['device_os_other'] = (df['device_os'] == 'other').astype('uint8')
        
        df['CA_BA'] = ((X['employment_status']=='CA') & (X['housing_status']=='BA')).astype('uint8')
        df['BA_windows'] = ((X['housing_status']=='BA') & (df['device_os'] == 'windows')).astype('uint8')
        
        df['intended_balcon_peak_1'] = ((df['intended_balcon_amount'] > 0 )
                                  & (df['intended_balcon_amount'] < 60)).astype('uint8')
        df['cur_address_months_binned_danger_spike'] = ((df['current_address_months_count'] > 25) &
                                                          (df['current_address_months_count'] <= 200))\
                                                              .astype('uint8') 
                                                              
        # рекодируем income по train
        df['income'] = df['income'].map(self.income_map).fillna(0).astype('uint8')
        
        # преобразовываем величины
        df['n_e_norm_similarity'] = self.quantile_similarity.transform(df[['name_email_similarity']])
        df['bank_months_count_qt'] = self.quantile_bank_months.transform(df[['bank_months_count']])
        df['cur_address'] = self.power_transformer.transform(df[['current_address_months_count']])
        df['zip_logged'] = np.log(df['zip_count_4w']+1)
        df['birth_distinct_emails_logged'] = np.log(df['date_of_birth_distinct_emails_4w']+1)
        
        # находим пиковое значение bank_branch_count
        labels = [f"q{i+1}" for i in range(len(self.bank_branch_bins) - 1)]
        
        bank_branch_count_q = pd.cut(X['bank_branch_count_8w'], bins=self.bank_branch_bins, 
                                        labels=labels, include_lowest=True)
        bank_branch_frame = pd.get_dummies(bank_branch_count_q).astype('uint8')

        q1 = bank_branch_frame.get('q1', pd.Series(0, index=df.index))
        q3 = bank_branch_frame.get('q3', pd.Series(0, index=df.index))
        
        if not q1.any() or q3.any():
            print('WARNING: слипшиеся квантили для bank_branch_count_8w')
        # создаем бинарный индикатор пикового значения
        df['bank_branch_peak'] = (q1 | q3).astype('uint8')
        
        # создаем фичи-формулы
        df['max_account_sum'] = df['bank_months_count'] * df['proposed_credit_limit']
        df['zip_transaction_ratio'] = df['zip_count_4w'] * 2 / (df['bank_branch_count_8w']+1)
        df['max_zip_sum'] = df['zip_count_4w'] * df['proposed_credit_limit']
        df['zip_sum_ratio'] = df['zip_count_4w'] / df['proposed_credit_limit']
        
        # берем две главные компоненты признаков-скоростей
        transformed_vel = self.pca_velocity.transform(df[self.vel_features])
        df['vel_1'], df['vel_2'] = transformed_vel[:, 0], transformed_vel[:, 1]
    
        # возвращаем итоговый датафрейм
        return df[self.features_list]