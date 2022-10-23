# DataAnalysisClassificationLab
Выполнили студенты 19ПМИ-1:
* **Никита Касьянов**
* **Егор Бузанов**

## Pipeline
Весь пайплайн прописан в _dvc.yaml_.

![image](https://user-images.githubusercontent.com/71206801/197386519-a6f13098-9761-41ba-bf98-ab47eabb6909.png)

## Experiments and metrics
Параметры для catboost находятся в файле _params.yaml_. В нашем случае, для примера, мы храним только один параметр _n\_estimators_, который мы можем менять при запуске эксперимента. Метрики сохраняем в файл _metrics.json_. Основная метрика для выбора лучшей модели была выбрана _fbeta_score_ с параметром _beta=2_, это означает, что эта метрика больше предпочтения отдаёт _recall'у_, а не _precision'у_. В этой задаче нам важнее _recall_ так, как мы не хотим правильно определять больше больных людей, потому что от этого зависят их жизни.    <br/>
Ниже приведён пример запуска эксперимента.

![image](https://user-images.githubusercontent.com/71206801/197386899-2a9e6183-00cc-4e6c-9ee6-d2a559351229.png)

![image](https://user-images.githubusercontent.com/71206801/197386914-dfa574d9-4d24-4edf-8d2a-d675ac8b057d.png)

