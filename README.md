# Создание признака индикатора
# 🧠 Feature Engineering Parser

Парсер условий для автоматического создания признаков в pandas DataFrame

## Что делает этот код?

Позволяет создавать новые столбцы в таблицах через текстовые условия вида:
- `"столбец > mean"` (значение больше среднего)
- `"столбец < median"` (значение меньше медианы) 
- `"столбец >= 5.0"` (значение больше или равно 5)

## Быстрый старт

```python
from feature_parser import create_feature_from_conditions

# Ваши данные
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000]
})

# Создаем признак: возраст выше среднего И зарплата выше медианной
conditions = [
    "age > mean",
    "salary > median"
]

result = create_feature_from_conditions(df, conditions, 'is_senior')

#не забудьте pip install -r requirments.txt 🐳🐳🐳
