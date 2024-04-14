# Start_DS_Inflation_Salaries
учебный проект по анализу заработных плат в России в курсе "Старт в Data Science!".
[описание курса](https://stepik.org/course/194633/info)

### Раздел: финальный проект

#### Часть 1
Скачайте данные о "Среднемесячной номинальной начисленной заработной плате работников организаций по видам экономической деятельности в Российской Федерации за 2000-2023 гг." по ссылке
Скачайте данные об уровне инфляции в стране по ссылке
Выберите 2-3 вида экономической деятельности, наиболее интересные Вам. Постройте графики изменения зарплаты по годам для этих видов экономической деятельности. Сделайте выводы
Пересчитайте средние зарплаты с учетом уровня инфляции и сравните, как влияет инфляция на изменение зарплаты по сравнению с предыдущим годом
Выберите подходящие визуализации и отобразите динамику изменения реальных зарплат с учетом инфляции. Сделайте выводы

#### Часть 2
Опционально: вынесите скаченные данные на публичный сервер (Neon, ElephantSQL, etc.)
Реализуйте веб-сервис, который загружает данные (локально или из внешнего хранилища) и предоставляет их соответствующее отображение
Добавьте в сервис визуализации данных из первой части проекта
На первом этапе всю аналитику и выводы выполняем в Jupyter Notebook; на втором этапе - переносим ее в Streamlit.
Итоговый результат - это подробная аналитика, представленная в виде веб-приложения, опубликованного на Streamlit Cloud.

#### Дополнительные исследования
Мы предложили Вам учесть инфляцию для подсчета реальной динамики зарплат. Также предлагаем Вам самостоятельно найти в открытых источниках важные экономические показатели для страны в разрезе по годам и изучить их корреляцию с динамикой реальных зарплат (с учетом инфляции). 

Результаты также отобразите в Streamlit-приложении.

### Файлы
[Ссылка на соревнование с датасетом](https://www.kaggle.com/competitions/playground-series-s4e4/data)
- streamlit_app.py: streamlit app
- salaries.csv: датасет с зарплатами
- inflation.csv: - датасет с инфляцией
- Other_char.xlsx: - датасет с дополнительными характеристиками
- requirements.txt: необходимые пакеты для работы приложения

🔭 [Ноутбук с кратким анализом данных](https://github.com/nedokormysh/Start_DS_Inflation_Salaries/blob/main/Start_DS_final.ipynb)

### Streamlit сервис
🔭 [Развёрнутый сервис](https://startdsinflationsalaries-14042024.streamlit.app/) 

### Автор 
* Илья Березуцкий
* t.me/nedokormysh
* e-mail: nedokormysh@live.com
