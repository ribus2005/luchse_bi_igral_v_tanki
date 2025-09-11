import pandas as pd
import numpy as np
import sklearn
print("версия sklearn", sklearn.__version__)

#загружаем датасетус
iris = sklearn.datasets.load_iris()

#делаем из датасетуса таблицу с данными
datasetus = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


datasetus['criterion'] = ((datasetus['sepal length (cm)'] > datasetus['sepal length (cm)'].mean()) & (datasetus['petal length (cm)'] > datasetus['petal length (cm)'].median()))


pd.set_option('display.max_rows', None)
print(datasetus)