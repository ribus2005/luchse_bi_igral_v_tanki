from packaging import version

import pandas as pd
import numpy as np
import sklearn


def check_version(package, ver):
    """
    проверяет версии пакетов
    """
    if version.parse(package.__version__) >= version.parse(ver):
        return True
    else:
        print(f"Версия слишком старая. Пожалуйста, обновите {package.__name__}")
        return False


def merge_columns(column1, column2):
    """
    выполняет поэлементное логическое и между столбцами
    """
    return column1 & column2


def set_display_params(paramlist):
    """
    меняет параметры вывода таблиц в pandas
    """
    for option, value in paramlist:
        pd.set_option(option, value)
    

def main():
    if not check_version(sklearn, "1.7.0"):
        return -1
    
    #загружаем датасетус
    iris = sklearn.datasets.load_iris()
    
    #делаем из iris dataset таблицу с данными
    datasetus = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    datasetus['species'] = iris.target
    
    
    #создаём новый столбец в таблице с названием criterion для булевого значения а, где a = True если длина чашелистика(sepal length) ириса больше средней и длина лепестка(petal length) больше медианы, иначе False
    
    #столбец по критерию длины чашелистика
    sepal_criterion = datasetus['sepal length (cm)'] > datasetus['sepal length (cm)'].mean()
    #столбец по критерию длины лепестка
    petal_criterion = datasetus['petal length (cm)'] > datasetus['petal length (cm)'].median()
    #объединяем столбцы, записываем в таблицу
    datasetus['criterion'] = merge_columns(sepal_criterion, petal_criterion)

    PrevMaxRows = pd.get_option('display.max_rows')
    PrevWidth = pd.get_option('display.width')
    #для вывода всей таблицы убираем лимит по количеству выводимых строк и увеличиваем ширину вывода
    set_display_params([('display.max_rows', None), ('display.width', 1000)])
    print(datasetus)
    #возвращаем параметры вывода обратно
    set_display_params([('display.max_rows', PrevMaxRows), ('display.width', PrevWidth)])
    return 0
    
if __name__ == "__main__":
    ret_val = main()
    if ret_val != 0:
        print(f"код ошибки {ret_val}")
