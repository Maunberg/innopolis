import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, auc
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier

pd.options.mode.chained_assignment = None
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow import feature_column
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from phik import phik_matrix
import pickle

RANDOM_STATE = 6033


def load(path, sep=None, dftype='csv'):
    data = None  # инициализируем переменную data
    if dftype == 'csv':  # если файл csv
        if sep is None:  # если не указан разделитель
            for i in [',', ';', '\t']:
                try:
                    data_new = pd.read_csv(path, sep=i)
                    data_new.columns
                    data = data_new.copy()
                    break  # выходим из цикла, если удалось прочитать файл
                except:
                    pass
        else:  # если указан разделитель
            try:
                data = pd.read_csv(path, sep=sep)
            except:
                return None
    elif dftype == 'xls':
        data = pd.read_excel(path)
    return data


def hasna(data):
    result = {}
    for i in data.columns:
        count = data[i].isna().sum()
        if count > 0:
            result[i] = count
    if len(result) > 0 and max(result.values()) / data.shape[0] < 0.05:
        result = {}
    return result


def autofillna(data, binar=False):
    result = hasna(data)
    if len(result) > 0:
        cat_columns = list(data.select_dtypes(include=['object', 'category']).columns)
        data[cat_columns] = data[cat_columns].fillna('Unknown')
        data.dropna(inplace=True)
        return data
    else:
        return data.dropna()


def duplcheck(data):
    data.drop_duplicates(inplace=True)
    for i in list(data.select_dtypes(include=['object', 'category']).columns):
        before = len(data[i].unique())
        sub = data[i].str.lower().replace(' ', '').replace('_', '')
        after = len(sub.unique())
        if before > after:
            data[i] = sub
    return data


def duplrecheck(data):
    for i in list(data.select_dtypes(include=['object', 'category']).columns):
        border = int(input('Выберите, сколько максимум категорий вы сможете распознать:'))
        categories = list(data[i].unique())
        if len(categories) <= border:
            print(
                'Пожалуйста, рассмотрите категории. Если среди них есть дубликаты, запишите их через запятую. Первым запишите категорию, на которую хотите заменить дубликаты. Если среди них нет дубликатов, то нажмите Enter:',
                *categories)
            sequence = input()
            if len(sequence) > 3:
                sequence = [j.strip() for j in sequence.split(',')]
                data[i] = data[i].replace(sequence[1:], sequence[0])
    return data


def duplcheck(data):
    data.drop_duplicates(inplace=True)
    for i in list(data.select_dtypes(include=['object', 'category']).columns):
        before = len(data[i].unique())
        sub = data[i].str.lower().replace(' ', '').replace('_', '')
        after = len(sub.unique())
        if before > after:
            data[i] = sub
    return data


def dtypes(data):
    errors = False
    for i in data.columns:
        check = {'num': 0,
                 'alpha': 0,
                 'decimal': 0}

        for j in range(len(data)):
            if str(data[i].iat[j]).isalpha():
                check['alpha'] += 1
            elif str(data[i].iat[j]).isnumeric():
                if str(data[i].iat[j]).count('.') > 0:
                    check['decimal'] += 1
                else:
                    check['num'] += 1

        if max(check) == 'num':
            if len(data) == check['num']:
                data[i] = data[i].astype('int')
            else:
                errors = True
        elif max(check) == 'alpha':
            if len(data) == check['alpha']:
                data[i] = data[i].astype('category')
            else:
                errors = True
        elif max(check) == 'decimal':
            if len(data) == check['dicimal']:
                data[i] = data[i].astype('float')
            else:
                errors = True
    return data, errors


def findcorr(data):
    corr_matrix = phik_matrix(data)
    corr_matrix = corr_matrix[(1 > corr_matrix) & (corr_matrix > 0.75)].dropna(axis=0, how='all').dropna(axis=1,
                                                                                                         how='all')
    high_corr_cols = corr_matrix.columns.tolist()

    if len(high_corr_cols) > 0:
        print(corr_matrix)
        coldel = input('Мы обнаружили корреляцию. Пожалуйста, посмотрите на таблицу и выберите один из коррелирующих столбцов. \
    \nПамятка: лучше удалять менее информативные столбцы. \
    \nЗапишите те столбцы, которые нужно удалить (если вы не хотите удалять коррелирующие столбцы, нажмите Enter):')
        if len(coldel) > 0:
            coldel = [j.strip() for j in coldel.split(',')]
            data = data.drop(coldel, axis=1)
        return data


def quant(column, data, figsize=(5, 5), error=False, ylim=False, show=True):
    if error == False:
        q1 = np.percentile(data[column], 25)
        q3 = np.percentile(data[column], 75)
    else:
        q3 = data[column].describe()['75%']
        q1 = data[column].describe()['25%']
    iqr = q3 - q1
    new_data = data[(data[column] > (q1 - 1.5 * iqr)) & (data[column] < (q3 + 1.5 * iqr))]
    if ylim != False:
        new_dataplot.set_ylim(ylim)
    if show == True:
        fig, ax = plt.subplots(figsize=(5, 5))
        new_dataplot = new_data.boxplot(column=column, figsize=figsize).set_title('Распределение ' + str(column))
        ax.set_ylabel('value')
        plt.show()
    if new_data.shape[0] / data.shape[0] > 0.75:
        return new_data
    else:
        return data


def analitic(data):
    num_cols = list(pd.DataFrame(data).select_dtypes(include=['int64', 'float64']).columns)
    new = data.copy()
    for i in num_cols:
        new = quant(i, new, show=False)
    num = 1 - new.shape[0] / data.shape[0]
    if new.shape[0] / data.shape[0] > 0.75:
        return new, False, num
    else:
        return new, True, num


def dtsplit(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(target, axis=1), data[target], train_size=0.75,
                                                        test_size=0.25, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def checkclasses(y_train):
    class_ratios = y_train.value_counts() / len(y_train)
    answer = min(class_ratios) / max(class_ratios)
    if answer < 0.5:
        return 'roc_auc'
    else:
        return 'f1'


def Randomiz_search(model, params, num_iter, cv_mod, X_train, y_train):
    metric = checkclasses(y_train)
    model = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=num_iter,
        verbose=200,
        scoring=metric,
        cv=KFold(cv_mod, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def model_search(X_train, X_test, y_train, y_test):
    ss = StandardScaler()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

    regressor_params = {
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__C': np.logspace(-4, 4, 20),
        'logisticregression__solver': ['liblinear', 'saga']
    }

    forest_params = {
        'randomforestclassifier__n_estimators': range(100, 301, 25),
        'randomforestclassifier__max_depth': range(5, 20, 2),
        'randomforestclassifier__min_samples_split': (3, 4, 5, 6),
        'randomforestclassifier__min_samples_leaf': (3, 4, 5, 6)
    }

    LGBM_params = {
        'max_depth': range(10, 41, 5),
        'learning_rate': [0.1, 0.01, 0.001]
    }

    XGBoost_params = {
        'gradientboostingclassifier__max_depth': range(10, 31, 5),
        'gradientboostingclassifier__learning_rate': [0.1, 0, 3, 0.01, 0.001],
        'gradientboostingclassifier__n_estimators': range(100, 301, 50)
    }

    CatBoost_params = {
        'learning_rate': [0.1, 0.01, 0.001],
        'depth': range(1, 30, 5),
        'n_estimators': range(100, 301, 50)
    }

    cat_cols = list(X_train.select_dtypes(include=['object', 'category']).columns)
    num_cols = list(X_train.select_dtypes(include=['int64', 'float64']).columns)

    col_transformer_ohe = make_column_transformer(
        (ohe, cat_cols),
        (ss, num_cols),
        remainder='passthrough'
    )

    col_transformer_enc = make_column_transformer(
        (enc, cat_cols),
        (ss, num_cols),
        remainder='passthrough'
    )

    pipeline_linear = make_pipeline(col_transformer_ohe, LogisticRegression(RANDOM_STATE))
    pipeline_forest = make_pipeline(col_transformer_ohe, RandomForestClassifier(random_state=RANDOM_STATE))
    pipeline_XGBoost = make_pipeline(col_transformer_ohe, GradientBoostingClassifier(random_state=RANDOM_STATE))

    Regression = Randomiz_search(pipeline_linear, regressor_params, 100, 10, X_train, y_train)
    results_data = pd.DataFrame({'model_name': ['LogisticRegression'], 'ROC_AUC': [round(Regression.best_score_, 4)]})

    Forest = Randomiz_search(pipeline_forest, forest_params, 100, 10, X_train, y_train)
    results_data = results_data.append(
        {'model_name': 'RandomForestClassifier', 'ROC_AUC': round(Forest.best_score_, 4)}, ignore_index=True)

    model_XGBoost = Randomiz_search(pipeline_XGBoost, XGBoost_params, 50, 15, X_train, y_train)
    results_data = results_data.append(
        {'model_name': 'GradientBoostingClassifier', 'ROC_AUC': round(model_XGBoost.best_score_, 4)}, ignore_index=True)

    model_LGBM = Randomiz_search(LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE), LGBM_params, 50,
                                 15, X_train, y_train)
    results_data = results_data.append({'model_name': 'LGBMClassifier', 'ROC_AUC': round(model_LGBM.best_score_, 4)},
                                       ignore_index=True)

    model_CBC = Randomiz_search(
        CatBoostClassifier(grow_policy='Depthwise', cat_features=cat_cols, random_state=RANDOM_STATE, verbose=False),
        CatBoost_params, 50, 15, X_train, y_train)
    results_data = results_data.append({'model_name': 'CatBoostClassifier', 'ROC_AUC': round(model_CBC.best_score_, 4)},
                                       ignore_index=True)

    model_name = results_data.sort_values('ROC_AUC').iloc[-1]['model_name']

    if model_name == 'LogisticRegression':
        best_model = Regression
    elif model_name == 'RandomForestClassifier':
        best_model = Forest
    elif model_name == 'GradientBoostingClassifier':
        best_model = model_XGBoost
    elif model_name == 'LGBMClassifier':
        best_model = model_LGBM
    elif model_name == 'CatBoostClassifier':
        best_model = model_CBC

    return best_model, results_data


def simple_search(X_train, X_test, y_train, y_test):
    LGBM_params = {
        'max_depth': range(10, 41, 5),
        'learning_rate': [0.1, 0.01, 0.001]
    }

    model_LGBM = Randomiz_search(LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE), LGBM_params, 50,
                                 15, X_train, y_train)

    return model_LGBM, round(model_LGBM.best_score_, 4)

