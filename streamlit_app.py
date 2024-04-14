import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Tuple, List


@st.cache_data
def open_data(path: str = 'salaries') -> pd.DataFrame:
    """
    Функция загружает датасет
    :param path: path to dataset
    :type path: str
    :return: pandas.DataFame
    """
    URL_SALARIES = 'data/salaries.csv'
    URL_INFLATION = 'data/inflation.csv'
    URL_ADD_INFO = 'data/Other_char.xlsx'

    if path == 'salaries':
        df = pd.read_csv(URL_SALARIES, index_col=0)
    elif path == 'inflation':
        df = pd.read_csv(URL_INFLATION, encoding='windows-1251', index_col=0)
    else:
        df = pd.read_excel(URL_ADD_INFO, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')

    print('Датасет загружен.')
    return df


def side_bar_params() -> Tuple[int, int]:
    """
    Функция формирования сайдбаров настройки размеров отображения графиков
    :return: кортеж с размерами
    """

    st.sidebar.markdown("*Настройки отображения матрицы корреляции*")

    width_cor: int = st.sidebar.slider("ширина таблицы корреляции", 2, 5, 5)
    height_cor: int = st.sidebar.slider("высота таблицы корреляции", 4, 7, 5)

    return width_cor, height_cor


# берём значения сайдбаров, которые будем использовать при отображении графиков
# height_corr, width_corr = side_bar_params()


def corr_matrix(all_data: pd.DataFrame,
                width: int = 5,
                height: int = 5) -> matplotlib.figure.Figure:
    """
    Функция отображения матрицы корреляции
    :param all_data: датафрейм
    :param width: ширина
    :param height: высота
    :return: график
    """

    fig = plt.figure(figsize=(width, height))
    sns.set_style("whitegrid")

    mask = np.triu(np.ones_like(all_data.corr(
        numeric_only=True
    ), dtype=bool))

    heatmap = sns.heatmap(all_data.corr(
        numeric_only=True
    ).round(2),
                          annot=True,
                          square=True,
                          cmap="BrBG",
                          cbar_kws={"fraction": 0.01},
                          linewidth=2,
                          mask=mask,
                          )

    heatmap.set_title("Треугольная тепловая карта корреляции Пирсона", fontdict={"fontsize": 11}, pad=5)

    return fig


def df_info(df: pd.DataFrame) -> None:
    """
    Функция выводит на экран информацию о пропусках и дублях в датасете
    :param df: датафрейм
    :return: None
    """
    st.text(('Есть пропуски!') if df.isna().any().any() else ('Пропусков нет'))
    st.text(('Есть дубли!') if df.duplicated().any() else ('Дубликатов нет'))
    st.text(f'Размер датасета {df.shape}')


def line_graph(df: pd.DataFrame,
               selected_options: List[str],
               start_year: int = 2000,
               stop_year: int = 2023) -> matplotlib.figure.Figure:
    """
    Функция отрисовки непрерывных графиков
    :param df: датафрейм
    :param selected_options: выбранные характеристики
    :param start_year: начальный год
    :param stop_year: конечный год
    :return: график для отрисовки
    """
    fig = plt.figure(figsize=(6, 3))

    colors = ['orange', 'red', 'green']
    # print(selected_options)

    df_temp = df.loc[selected_options, start_year: stop_year]
    # Отображение графиков
    for i, row in df_temp.iterrows():
        plt.plot(row.index,
                 row.values,
                 color=colors[df_temp.index.get_loc(i)],
                 label=i)
    
    plt.title('Изменение среднемесячной номинальной заработной платы\n без учёта инфляции')
    plt.xlabel('Год')
    plt.ylabel('Среднемесячная заработная плата (руб)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(rotation=45)

    return fig

def hist_graph(df: pd.DataFrame,
               selected_option: List[str]) -> matplotlib.figure.Figure:
    """
    Функция отрисовки гистрограмм для непрерывных величин
    :param df: датафрейм
    :param selected_option: выбранные признаки
    :return: график для отрисовки
    """
    fig = plt.figure(figsize=(6, 3))
    # Выбор данных по отрасли
    data = df.loc[selected_option]

    mean = data.mean()
    median = data.median()

    st.write(f'Разница между средним и медианным значением'
             f' в {selected_option} = {mean - median}')

    # Отображение гистограммы
    sns.histplot(data=df.loc[selected_option],
                 kde=True,
                 bins=10,
                 color='gray',
                 edgecolor='black',
                 linewidth=1
                 )

    # Настройка оформления графика
    plt.title(f'Распределение зарплат в {selected_option}')
    plt.xlabel('Зарплата (руб)')
    plt.grid(True)

    # Добавление среднего и медианного значений на график
    plt.axvline(mean, color='red', linestyle='--', label='Среднее значение')
    plt.axvline(median, color='green', linestyle='-.', label='Медианное значение')
    plt.legend()

    return fig


def inflation_graph(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Функция отрисовки гистрограмм для непрерывных величин
    :param df: датафрейм инфляции
    :return: график для отрисовки
    """
    fig = plt.figure(figsize=(6, 3))
    inflation_year_data = df['Всего']

    # Создание бар-графика
    colors = cm.rainbow(np.linspace(0, 1, len(inflation_year_data)))
    plt.bar(inflation_year_data.index, inflation_year_data.values, color=colors)

    plt.plot(inflation_year_data.index, inflation_year_data.values, color="darkgray")

    # Настройка оформления графика
    plt.title('Общая инфляция за год')
    plt.xlabel('Год')
    plt.ylabel('Процент')
    plt.grid(True)

    return fig


def real_income(salaries: pd.DataFrame,
                inflation: pd.DataFrame) -> pd.DataFrame:
    salaries_real = pd.DataFrame(index=salaries.index, columns=salaries.columns)

    # вычисляем реальные зарплаты для каждого года и вида деятельности
    for year in salaries.columns:
        inflation_rate = inflation.loc[year, 'Всего'] / 100
        salaries_real[year] = salaries[year] / (1 + inflation_rate)

    return salaries_real

def plot_salaries(salaries: pd.DataFrame,
                  salaries_real: pd.DataFrame,
                  inflation: pd.DataFrame,
                  activities: list,
                  start_year: int = 2000,
                  stop_year: int = 2023,
                  verbose_inflation: bool = True) -> None:
    """
    Функция отрисовывает графики зарплат с отображением номинальной и реальной
    заработной платы для трёх видов деятельности
    :param salaries: датафрейм зарплат
    :param salaries_real: датафрейм реальных зарплат
    :param inflation: датафрейм инфляции
    :param activities: выбранные признаки
    :param start_year: начальный год
    :param stop_year: конечный год
    :param verbose_inflation: режим отображения инфляции
    :return: None
    """
    salaries = salaries.loc[:, start_year: stop_year]
    # st.dataframe(salaries)
    salaries_real = salaries_real.loc[:, start_year: stop_year]
    inflation = inflation.loc[stop_year: start_year]

    for activity in activities:
        fig, ax1 = plt.subplots(figsize=(8, 6))

        color = 'tab:brown'
        ax1.set_xlabel('Год')
        ax1.set_ylabel('Зарплата', color=color)
        ax1.plot(salaries.columns, salaries.loc[activity],
                 marker='*',
                 label='Номинальная зарплата',
                 color='red')
        ax1.plot(salaries_real.columns, salaries_real.loc[activity],
                 marker='x',
                 label='Реальная зарплата',
                 color='orange')
        ax1.bar(salaries.columns,
                salaries.loc[activity] - salaries_real.loc[activity],
                label='Дельта',
                color='darkgrey',
                # bottom=salaries_real.loc[activity]
                )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(range(start_year, stop_year + 1))
        ax1.set_xticklabels(range(start_year, stop_year + 1), rotation=45)

        if verbose_inflation:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Инфляция, %', color=color)
            ax2.plot(inflation.index, inflation['Всего'],
                 label='Инфляция',
                 marker='+',
                 color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        plt.grid(True)
        plt.title(activity)
        ax1.legend(loc='upper center')
        if verbose_inflation:
            ax2.legend(loc='upper right')
        # plt.show()

        st.pyplot(fig)

        nominal = salaries.loc[activity]
        real = salaries_real.loc[activity]
        correlation = nominal.corr(real)
        st.text(f"Корреляция между зарплатами для {activity}: {correlation:.6f}")

def norm_graph(norm_df: pd.DataFrame,
               selected_options_areas,
               start_year_norm: int,
               stop_year_norm: int) -> matplotlib.figure.Figure:
    """
    Фунцкия отрисовывает графики нормированных значений характеристик
    :param norm_df: датафрейм с нормированными значениями
    :param selected_options_areas: выбранные характеристики для отображения
    :param start_year_norm: начальный год
    :param stop_year_norm: конечный год
    :return: -> matplotlib figure для отображения streamlit
    """
    fig = plt.figure(figsize=(6, 5))


    norm_df = norm_df.loc[start_year_norm: stop_year_norm]
    # st.dataframe(norm_df)
    # st.write(norm_df.index)

    # добавляем линии для каждой выбранной характеристики
    for option in selected_options_areas:
        plt.plot(norm_df.index, norm_df[option], label=option)

    # добавляем легенду
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(range(start_year_norm, stop_year_norm + 1))
    plt.xticks(rotation=45)
    # добавляем подписи осей
    plt.xlabel('Год')
    plt.title('Графики нормированных значений')

    return fig


def process_main_page():
    show_main_page()
    # определяем значения из параметров сайдбаров
    height_corr, width_corr = side_bar_params()

    # Загрузка данных
    with st.spinner('Пожалуйста, подождите. Идёт загрузка данных'):
        salaries = open_data('salaries')
        salaries.columns = salaries.columns.astype(int)
        inflation = open_data('inflation')
        inflation.dropna(inplace=True)
        inflation = inflation.loc[2023:2000]

        other_info = open_data('other')
        salaries_real = real_income(salaries, inflation)
    st.success('Все датасеты загружены!')


    st.title('Краткий обзор изменения заработных плат с 2000 до 2023 год')

    st.markdown("**Сэмплы загруженных датасетов**")

    st.markdown('### Датасет зарплат')
    st.write('Среднемесячная номинальная начисленная заработная'
             ' плата работников организаций по видам экономической деятельности'
             ' в Российской Федерации за 2000-2023 гг.')
    st.dataframe(salaries.sample(3))

    df_info(salaries)

    st.markdown('### Датасет инфляции')
    st.write('Таблица уровня инфляции по месяцам в годовом исчислении')
    st.dataframe(inflation.sample(3))

    df_info(inflation)

    # st.text('В данных об инфляции у нас есть пропущенные значения\n'
    #         'Не будем рассматривать эти годы\n'
    #         'Также нас интересуют данные только с 2000 года')

    st.markdown('### Датасет дополнительных показателей')
    st.text('- Кол-во безработных в РФ, в % \n'
            '- ВВП\n'
            '- Индекс счастья\n'
            '- Численность населения\n'
            '- СКР')
    st.dataframe(other_info.sample(3))

    df_info(other_info)

    st.markdown('---')

    st.markdown('## Графики')

    st.markdown('### Графики номинальной заработной платы')

    # Создаем список вариантов выбора
    activities = ['рыболовство_и_рыбоводство',
                  'строительство',
                  'добыча_полезных_ископаемых']

    # Добавляем контрол выбора
    selected_options = st.multiselect(':blue[Выберите графики]',
                                      activities,
                                      default=activities)

    # Создаем ползунок от 2000 до 2023 года
    start_year, end_year = st.slider(
        ':blue[Выберите диапазон лет:]',
        min_value=2000,
        max_value=2023,
        value=[2000, 2023]
    )

    # Отображаем выбранный диапазон лет
    st.write('Выбранный диапазон лет:', start_year, 'до', end_year)

    st.pyplot(line_graph(salaries, selected_options, start_year, end_year),
              use_container_width=True)

    st.markdown('**Выводы:**\n- Во всех трёх видах деятельности наблюдался рост'
                '\n- Из интересных особенностей - изменение наклона роста зарплат в рыболовстве и рыбоводстве')

    st.info('Для выбранных отраслей можно посмореть гистограммы распределений')

    if st.button(':rainbow[Раскрыть графики распределений]'):
        st.balloons()

        for act in selected_options:
            st.pyplot(hist_graph(salaries, act),
                      use_container_width=True)

        st.markdown('**Выводы:**\n- везде скошенное вправо распределение'
                    '\n- максимальная разница между средним и медианным значением в "рыболовстве и рыбоводстве"')

    st.markdown('### Графики инфляции')

    st.pyplot(inflation_graph(inflation))

    st.markdown('**Выводы:**\n'
                '- Стабильной тенденции по инфляции я не вижу.'
                '- Но всё таки в начале 2000 инфляция была максимальной')

    st.markdown('### Графики реальной заработной платы')

    # Создаем список вариантов выбора
    activities_real = ['рыболовство_и_рыбоводство',
                       'строительство',
                       'добыча_полезных_ископаемых']

    # Добавляем контрол выбора
    selected_options_real = st.multiselect(':blue[Выберите отрасли]',
                                            activities_real,
                                            default=activities_real)

    # Создаем ползунок от 2000 до 2023 года
    start_year_real, end_year_real = st.slider(
        ':blue[Выберите диапазон лет для отображения:]',
        min_value=2000,
        max_value=2023,
        value=[2000, 2023]
    )

    # Отображаем выбранный диапазон лет
    st.write('Выбранный диапазон лет:', start_year_real, 'до', end_year_real)

    # выбираем режим отображения инфляции на графике
    verbose_inf = st.checkbox(':blue[Отобразить изменение инфляции на графиках]')
    # отображаем графики реальной заработной платы
    plot_salaries(salaries,
                  salaries_real,
                  inflation,
                  selected_options_real,
                  start_year_real,
                  end_year_real,
                  verbose_inflation=verbose_inf)

    st.markdown('**Выводы:**\n- Мы имеем большую корреляцию между реальной и номиальной зарплатами,'
                'не существенно, но всё таки в рыбоводстве и рыболовстве эта корреляция чуть выше'
                '\n- Для рыбоводства и рыболовства, пожалуй, не видно явного влияния инфляции. '
                'Но для других сфер деятельности можно говорить, что при росте инфляции в 2014-2015 годах рост'
                ' реальной заработной платы замедлялся')

    st.markdown('---')

    st.markdown('## Дополнительные исследования')

    st.markdown('Были выбраны следующие показатели:\n'
                '- Кол-во безработных в РФ, в % к экономически активному населению (рабочей силе)\n'
                '- Валовой внутренний продукт (в текущих ценах, млрд.руб.)\n'
                '- Индекс счастья\n'
                '- Численность населения с денежными доходами ниже границы бедности/величины прожиточного минимума\n'
                '- Суммарный коэффициент рождаемости (СКР) (всё население)')

    df_all = pd.concat([salaries_real, other_info], axis=0)
    df_all = df_all.T

    st.info('Размеры матрицы корреляции можно менять благодаря слайдерам слева\n'
            'Ширина отображения ограничена и подстраивается под ширину страницы')
    st.pyplot(corr_matrix(df_all, width=width_corr, height=height_corr))

    st.markdown('**Выводы**\n'
                '- все выбранные характеристики (пожалуй кроме СКР)'
                ' имеют сильную линейную взаимосвязь с реальными зарплатами\n'
                '- уровень безработицы имеет слабую корреляцию с СКР и индексом счастья\n'
                '- ВВП чуть хуже линейно связан с СКР и индексом счастья\n'
                '- индекс счастья меньше всего коррелирует с уровнем бедности и СКР'
                '- уровень бедности вообще не имеет линейной корреляции только с индексом счастья')

    st.markdown('### Нормированные графики')

    st.text('Для изучения графиков зависимостей отнормируем значения характеристик'
            ' к максимальному значению для отображения на едином графике')
    norm_df = df_all.copy()  # создаем копию датафрейма

    for col in norm_df.columns:
        max_value = norm_df[col].max()
        norm_df[col] = norm_df[col] / max_value

    norm_df.index = pd.to_numeric(norm_df.index)

    # Создаем список вариантов выбора
    areas = ['рыболовство_и_рыбоводство',
             'строительство',
             'добыча_полезных_ископаемых',
             'уровень_безработицы',
             'ВВП',
             'индекс_счастья',
             'уровень_бедности',
             'СКР']

    # Добавляем контрол выбора
    selected_options_areas = st.multiselect(':blue[Выберите графики для отображения]',
                                           areas,
                                           default=areas)

    # Создаем ползунок от 2000 до 2023 года
    start_year_norm, end_year_norm = st.slider(
        ':blue[Выберите года отображения:]',
        min_value=2000,
        max_value=2023,
        value=[2000, 2023]
    )
    # Отображаем выбранный диапазон лет
    st.write('Выбранный диапазон лет:', start_year_norm, 'до', end_year_norm)

    st.pyplot(norm_graph(norm_df=norm_df,
                         selected_options_areas=selected_options_areas,
                         start_year_norm=start_year_norm,
                         stop_year_norm=end_year_norm))
def show_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Обзор изменения заработных плат с 2000 до 2023 год",
        # page_icon=image
    )


process_main_page()
