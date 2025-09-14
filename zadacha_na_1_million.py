import pandas as pd
import numpy as np
import sklearn
#должна быть не ниже 1.7.0
print("версия sklearn", sklearn.__version__)

#загружаем датасетус
iris = sklearn.datasets.load_iris()

#делаем из датасетуса таблицу с данными
datasetus = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

#создаём новый столбец в таблице с названием criterion для булевого значения а, где a = True если длина чашелистика(sepal length) ириса больше средней и длина лепестка(petal length) больше медианы, иначе False

#столбец по критерию длины чашелистика
sepal_criterion = datasetus['sepal length (cm)'] > datasetus['sepal length (cm)'].mean()
#столбец по критерию длины лепестка
petal_criterion = datasetus['petal length (cm)'] > datasetus['petal length (cm)'].median()
#объединяем столбцы, записываем в таблицу
datasetus['criterion'] = sepal_criterion & petal_criterion

#для вывода всей таблицы убираем лимит по количеству выводимых строк
pd.set_option('display.max_rows', None)

print(datasetus)
