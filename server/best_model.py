from pipe_layers import *
from utils.support import *
import catboost
from sklearn.pipeline import Pipeline

# загружаем данные
lin_features = open_list('../logs/top_linear_features.txt')

data = pd.read_csv('../data/bank_data.csv', sep=',')
data_test = open_data('../data/bank_test.csv')

best_model = open_model('../logs/cat_boost.pkl')
best_params = best_model.get_params()

# разделяем данные на тренировочную и тестовую выборки
train_inds = data.index.difference(data_test.index)
data_train = data.loc[train_inds]
data_test = data.loc[data_test.index]

X_test, y_test = data_test.drop('fraud_bool', axis=1), data_test['fraud_bool']
X_train, y_train = data_train.drop('fraud_bool', axis=1), data_train['fraud_bool']
# создаем и обучаем pipeline
pipe = Pipeline([
    ('MissingValuesPreprocessor', MissingValuesPreprocessor()),
    ('BankMonthsInputer', BankMonthsInputer()),
    ('FeatureEngineering', FeatureEngineering(lin_features)),
    ('CatBoost', catboost.CatBoostClassifier(**best_params))
])

pipe.fit(X_train, y_train)

# сохраняем модель
with open("./final_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)