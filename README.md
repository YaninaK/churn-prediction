# Предсказание оттока клиентов

## 1. Данные

* Исходные данные находятся [по ссылке](https://drive.google.com/file/d/1TAVECAfnel9lPfcpfel6qXhZSW2yNqdX/view?usp=sharing).

## 2. Методика подготовки данных

###  2.1. Формирование датасета

Из датасета исключены признаки noadditionallines и year - это постоянные величины.

Данные представлены за 3 месяца, но значения целевой функции не изменяется во времени. На заданном промежутке времени варьируют только признаки totalcallduration и avgcallduration.

Доступны данные по 9525 клиентам. Выборка несбалансированна: отток клиентов составляет 9%.

Разбивку на обучающую и тестовую выборки будем осуществлять по уникальным идентификационным номерам клиентов с сохранением пропорции между значениями целевой переменной ([train_test_data_split](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/data/validation.py)).

Создадим датасет, где в качестве индекса будет выступать id клиента ([generate_dataset](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/data/make_dataset.py)).

В среднем, по клиентам доступны данные только за 2 временных промежутка: 81.5% случаев оттока приходится на временной промежуток 2 месяца, 17.4% - 3 месяца.

По 84% клиентов отсутствует информация за третий месяц, причем более 91% из них обозначены как действующие клиенты (не отток).
Кроме того, по 0.5% клиентов отсутствует информация за первый месяц, это могли быть новые клиенты.

Создадим последовательности из трех периодов, дополнив недостающюю информацию нулями (padding) для признаков totalcallduration и avgcallduration. Последовательности в дальнейшем будут использоваться как входящие данные для блока LSTM.

Из датасета исключаем признак month.

Обучающий датасет, в свою очередь, делим на обучающий и валидационный.

### 2.2. Отбор признаков

Notebook c разведочным анализом данных и отбором признаков находится [по ссылке](https://github.com/YaninaK/churn-prediction/blob/main/notebooks/01_EDA_and_Feature_selection.ipynb).

Отбор признаков производился с помощью статистических методов (тест хи-квадрат, критерий Колмогорова — Смирнова).
Также для ориентира использовлись Recursive feature elimination (RFE) и Random Forest feature importances.

Для временных признаков totalcallduration и avgcallduration были сгенерированны ембеддинги с помощью [модели LSTM](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/LSTM_model.py)

Для признака state были сгенерированы ембеддинги с помощью tensorflow.keras.layers.StringLookup, tensorflow.keras.layers.Embedding и полносвязных слоев нейросетевой модели ([fit_transform_embeddings, transform_embeddings](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/embeddings_tf.py)).

Признаки education и occupation были закодированы с помощью встроенной функции one_hot в tensorflow.keras.layers.StringLookup ([fit_transform_one_hot_encoding, transform_one_hot_encoding](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/embeddings_tf.py)).


* Признаки age, numberofcomplaints и unpaidbalance ожидаемо отобрались и статистическим методами, и деревянными моделями.

* Согласно отбору RFE и Random Forest feature importances, эмбеддинги признаков, изменяющихся в времени - totalcallduration и avgcallduration и их производные также получили достаточно высокий рейтинг. Это послужило мотивацией для использования блока LSTM в нейросетевой модели.

* Ембеддинги state оказались в середине рейтинга RFE и не отобрались Random Forest, но у них есть потенциал в сочетании с временными ембеддингами: средний процент оттока существенно отличается по штатам от 2.7% (DE) до 16.1% (SC)

* Признак education имеет смысл использовать через one-hot encoding: самая большая разница в проценте оттока мажду PhD (10%) и Master (7.3%), в то время как High School (8.9%) и Bachelor (9.2%) неожиданно находятся между ними.

* Из дискретных признаков определенный потенциал имеет callfailurerate: разница оттока между 0.01 (8.1%) и 0.02 (9.5%) может улучшить результаты нейросетевой модели. Он был отобран деревянными моделями.

* Бинарные признаки maritalstatus и usesvoiceservice были отобраны деревянными моделями. К ним были добавлены customersuspended, gender, homeowner и usesinternetservice, у которых есть потенциал для улучшения результата нейросети - значиамая разница в проценте оттока между 0 и 1. Бинарные признаки, как правило, недооцениваются деревянными моделями.

* Непрерывные признаки annualincome, callingnum, monthlybilledamount, numdayscontractequipmentplanexpiring, penaltytoswitch, percentagecalloutsidenetwork, totalminsusedinlastmonth не прошли отбор с помощью критерия Колмогорова — Смирнова и не выглядят достаточно привлекательными, хотя и отобрались деревянными моделями.

* Baseline AUC для модели оттока составил 0.7055.


### 2.3. Трансформация признаков

Процесс подготовки данных предсталавлен в функции [data_preprocessing_pipeline](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/train.py)


#### 2.3.1. Трансформация признаков, изменяющихся на заданном временном промежутке

Для использования в LSTM блоке модели, последовательности нормализуются и приводятся к размерности (-1, 3, 2) ([fit_transform_seq, transform_seq](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/features/time_series_features.py)):

* 3 - длина последовательности (3 месяца),
* 2 - число признаков (totalcallduration и avgcallduration)


#### 2.3.2 Категориальные признаки

* Ембеддинги - признак state

Строковые признаки кодируются в [нейросетевой модели](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/LSTM_embeddings_model.py) с помощью модуля tensorflow.keras.layers.StringLookup.

Для ембеддингов используется модуль tensorflow.keras.layers.Embedding.

* One-hot encoding - признаки education и occupation

One-hot encoding выполняется с помощью встроенной функции в tensorflow.keras.layers.StringLookup.

* Бинарные признаки

Из бинарных признаков отобраны  customersuspended, gender, homeowner, maritalstatus, usesinternetservice и usesvoiceservice.

Бинарные категориальные признаки кодируются функцией [map_categorical_features](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/features/binary_features.py)


#### 2.3.3. Численные признаки

Из численных признаков отобраны callfailurerate, numberofcomplaints, age и unpaidbalance.

Отобранные признаки нормализуются с помощью sklearn.StandardScaler


## 3. Модель прогнозирования оттока клиентов

Notebook с базовой нейросетевой моделью прогнозирования оттока клиентов находится [по ссылке](https://github.com/YaninaK/churn-prediction/blob/main/notebooks/02_Baseline_model_NN.ipynb)

### 3.1.  Методика построения модели.

[Модель](https://github.com/YaninaK/churn-prediction/blob/main/src/churn_prediction/models/LSTM_embeddings_model.py) имеет 5 входов:

* Вход для блока LSTM, размерностью (-1, 3, 2): 
  - 3 - длина последовательности (3 месяца),
  - 2 - число признаков (totalcallduration и avgcallduration)

* Вход для получения ембеддингов признака state

* По одному входу для one-hot encoding признаков education и occupation

* Вход для остальных отoбранных признаков:
  - категориальные 

    * customersuspended,
    * gender,
    * homeowner,
    * maritalstatus,
    * usesinternetservice,
    * usesvoiceservice

  - численные

    * callfailurerate,
    * numberofcomplaints,
    * age,
    * unpaidbalance    


Строковые переменные кодируются с помощью модуля tensorflow.keras.layers.StringLookup.

Для ембеддингов используется модуль tensorflow.keras.layers.Embedding

One-hot encoding выполняется с помощью встроенной функции в tensorflow.keras.layers.StringLookup.


* Выход из блока LSTM конкатенируется с пропущенными через полносвязные слои остальными блоками.
* Объединенный блок еще раз пропускается через полносвязный слой, после чего следует финальный слой.
* В качестве оптимизатора используется tensorflow.keras.optimizers.Adam()
* Функция потерь - tensorflow.keras.losses.BinaryCrossentropy()

### 3.2. Оценка качества модели

Результаты модели на тестовой выборке:

* loss :  0.5514865517616272
* cross entropy :  0.5514865517616272
* Brier score :  0.19421622157096863
* tp :  143.0
* fp :  563.0
* tn :  1122.0
* fn :  39.0
* accuracy :  0.6775575876235962
* precision :  0.20254957675933838
* recall :  0.7857142686843872
* auc :  0.805473268032074
* prc :  0.2794499695301056

### 3.3. Анализ полученных результатов

* AUC на тестовой выборке 0.8055, что значительно выше, чем в базовом варианте Random Forest 0.7055
* ROC - кривая на обучающей и тестовой выборках практически совпадает.
* Precision-Recall кривые на обучающей и тестовой выборках достаточно близки.
* У модели неплохой лифт. На обучающей и тестовой выборке лифт модели практически совпадает.


## 4. Inference piplene

Inference piplene представлена в двух вариантах:

* [Notebook](https://github.com/YaninaK/churn-prediction/blob/main/notebooks/03_Inference_pipeline_nn.ipynb) c конвейером для инференса.
* [Файл](https://github.com/YaninaK/churn-prediction/blob/main/scripts/inference_script.py) c конвейером инференса для запуска из командной строки.

## 5. Оценка практической применимости модели

Модель имеет потенциал, AUC на тестовой выборке 0.8055 - неплохое начало.

Прежде чем применяять модель на практике необходимо выяснить причины пропуска данных:
* По 84% клиентов отсутствует информация за третий месяц, причем более 91% из них обозначены как действующие клиенты (не отток).

Если данные найдутся, имеет смысл переобучить модель на полном наборе.

Можно продлить длину последовательности. Например, не 3, а более месяцев. Таким образом, профиль клиента получится более объемный и есть шанс повысить точность модели.

Следует проанализировать, с какими группами клиентов модель работает лучше и сконцентрироваться на них, чтобы оптимизировать расходы на обработку False Positive результатов. 
С другой стороны, в развитие модели можно поискать решения для групп клиентов, где модель ошибается.

Слабое место нейросетевой модели - отсутствие интерпретируемости. Другие стейкхолдеры, скорее всего, захотят узнать, на основании каких факторов модель классифицирует клиентов.

Если в приоритете будет интерпретируемость, а не точность предсказания, можно использовать бустинг, например, GLM + Random Forest. Линейная модель (GLM) поможет оценть факторы, влияющие на результат, а Random Forest в бустинге уменьшит смещение модели и таким образом добавит точности. 
Для повышения точности, можно попробовать другие методы ансамблирования.