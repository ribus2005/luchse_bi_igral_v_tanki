    from packaging import version
    import pandas as pd
    import numpy as np
    import sklearn
    import re
    from typing import Union, List, Callable

    def check_version(package, ver):
        """Проверяет версии пакетов"""
        if version.parse(package.__version__) >= version.parse(ver):
            return True
        else:
            print(f"Версия слишком старая. Пожалуйста, обновите {package.__name__}")
            return False

    def set_display_params(paramlist):
        """Меняет параметры вывода таблиц в pandas"""
        for option, value in paramlist:
            pd.set_option(option, value)

    class ConditionParser:
        """Парсер условий для создания признаков из столбцов DataFrame"""
        
        OPERATIONS = {
            '>': lambda x, y: x > y,
            '>=': lambda x, y: x >= y,
            '<': lambda x, y: x < y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            '&': lambda x, y: x & y,
            '|': lambda x, y: x | y,
        }
        
        AGG_FUNCTIONS = {
            'mean': lambda x: x.mean(),
            'median': lambda x: x.median(),
            'std': lambda x: x.std(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'sum': lambda x: x.sum(),
        }
        
        @classmethod
        def parse_condition(cls, condition_str: str) -> Callable:
            """
            Парсит строку условия и возвращает функцию для применения к DataFrame
            Принимает имена численных столбцов, операторы сравнения 
            """
            pattern = r'(.+?)\s*(>|>=|<|<=|==|!=|&|\|)\s*(.+)'
            match = re.match(pattern, condition_str.strip())
            
            if not match:
                raise ValueError(f"Некорректный формат условия: {condition_str}")
            
            column_name = match.group(1).strip()
            operator = match.group(2).strip()
            value_str = match.group(3).strip()
            
            if operator not in cls.OPERATIONS:
                raise ValueError(f"Неподдерживаемая операция: {operator}")
            
            def condition_function(df: pd.DataFrame) -> pd.Series:
                if column_name not in df.columns:
                    raise ValueError(f"Столбец '{column_name}' не найден в DataFrame")
                
                if not pd.api.types.is_numeric_dtype(df[column_name]):
                    raise ValueError(f"Столбец '{column_name}' должен быть числовым")
                
                column_data = df[column_name]
                
                if value_str in cls.AGG_FUNCTIONS:
                    value = cls.AGG_FUNCTIONS[value_str](column_data)
                else:
                    try:
                        value = float(value_str)
                    except ValueError:
                        raise ValueError(f"Некорректное значение: {value_str}")
                
                return cls.OPERATIONS[operator](column_data, value)
            
            return condition_function

    def create_feature(
        df: pd.DataFrame, 
        conditions: Union[str, List[str]], 
        new_column_name: str = 'criterion'
    ) -> pd.DataFrame:
        """Создает новый признак на основе условий"""
        if isinstance(conditions, str):
            conditions = [conditions]
        
        parser = ConditionParser()
        condition_results = []
        
        for condition_str in conditions:
            condition_func = parser.parse_condition(condition_str)
            result_series = condition_func(df)
            condition_results.append(result_series)
        
        # Объединяем условия через логическое И
        combined_result = condition_results[0]
        for i in range(1, len(condition_results)):
            combined_result = combined_result & condition_results[i]
        
        result_df = df.copy()
        result_df[new_column_name] = combined_result
        
        return result_df

    def load_iris_dataset():
        """Загружает и подготавливает iris dataset"""
        iris = sklearn.datasets.load_iris()
        dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        dataset['species'] = iris.target
        return dataset

    def display_full_table(df):
        """Выводит полную таблицу без ограничений"""
        prev_params = [
            ('display.max_rows', pd.get_option('display.max_rows')),
            ('display.width', pd.get_option('display.width'))
        ]
        
        set_display_params([('display.max_rows', None), ('display.width', 1000)])
        print(df)
        
        # Восстанавливаем параметры
        for option, value in prev_params:
            pd.set_option(option, value)

    def main():
        """main содержит пример использования функций решения задачи, вы можете вызывать их со своими аргументами"""
        if not check_version(sklearn, "1.7.0"):
            return -1
        
        try:
            # Загружаем датасет
            dataset = load_iris_dataset()
            
            # Создаем признак по условиям
            conditions = [
                "sepal length (cm) > mean",
                "petal length (cm) > median"
            ]
            
            dataset_with_feature = create_feature(
                df=dataset,
                conditions=conditions,
                new_column_name='criterion'
            )
            
            # Выводим полную таблицу
            display_full_table(dataset_with_feature)
            
            return 0
            
        except Exception as e:
            print(f'Ошибка: {e}')
            return -2

    if __name__ == "__main__":
        ret_val = main()
        if ret_val != 0:
            print(f"Код ошибки: {ret_val}")
