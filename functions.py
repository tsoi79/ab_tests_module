import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from IPython.core.display import display, HTML
from collections import namedtuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def data_loader (
    path_to_groups='groups.csv',
    path_to_active='active_studs.csv',
    path_to_checks='checks.csv',
    path_to_add='',
    upload=False,
    path=''
    ):
    '''
        функция data_loader выполняет загрузку файлов, содержащих результаты А/В тестирования и формирует из них
        один pandas-датафрейм. Проводится проверка на отсутствие NaN-значений и уникальность id пользователей.
        ---
        Параметры:
            path_to_groups : str, default 'groups.csv' 
                путь к файлу формата csv, содержащему данные о распределении пользователей по группам.
            path_to_active : str, default 'active_studs.csv'
                путь к файлу формата csv, содержащему id пользователей, заходивших на сайт в период эксперимента.
            path_to_checks : str, default 'checks.csv'
                путь к файлу формата csv, содержащему данные об оплатах.
            path_to_add    : str, default ''
                путь к файлу формата csv, содержащему дополнительные данные о распределении по группам.
            upload         : boolean, default False
                параметр, определяющий, выгружать ли датафрейм groups в по указанному пользователем пути. 
                Если upload=True - происходит выгрузка в файл csv в директорию, указанную в параметре path.
            path           : str, default ''
                путь, по которому нужно выгрузить файл с содержимым датафрейма groups
                
        Важная информация: файлы должны быть в кодировке utf-8.
        ---
        Результат: pandas.DataFrame
            Датафрейм, содержащий информацию только о тех пользователей, 
            которые заходили на сайт во время эксперимента.
            Столбцы датафрейма: id(int64), grp(str), rev(float64)
                
    '''
    
    # используемые функции:
    
    def read_csv_file(path):
        '''
            функция read_csv_file считывает данные из файла csv в датафрейм. 
            Разделитель определяется автоматически, кодировка используется utf-8.
            ---
            Параметры:
                path - путь к месту хранения файла.
            Результат:
                датафрейм, сформированный из файла csv.
        '''
        reader = pd.read_csv(path, engine='python', sep = None, iterator = True)
        infer_sep = reader._engine.data.dialect.delimiter
        res_df = pd.read_csv(path, sep = infer_sep)
        return res_df
    
    def nan_checking(data):
        '''
            функция nan_checking проверяет датафрейм на наличие значений NaN.            
            ---
            Параметры:
                data - датафрейм для проверки.
            Результат:
                Если NaN нет - то функция возвращает False.
                Если есть хотя бы одно такое значение, функция возвращает True.
        '''
        if data.isna().any().any():
            return False
        else:
            return True
    
    def is_unique(cheking_series):
        '''
            Функция is_unique проверяет серию на уникальность.
            Сравнивает количество записей в серии с количеством уникальных значений.
            ---
            Параметры:
                cheking_series - проверяемая серия.
            Результат:
                False - в серии есть повторяющиеся значения.
                True - в серии все значения уникальны.
            
        '''
        a = cheking_series.count()
        b = cheking_series.nunique()
        if a != b:
            return False
        else:
            return True
        
    #----- Подготовка данных -----
    
    # cчитывание данных в датафреймы:
    groups = read_csv_file(path_to_groups)
    active_studs = read_csv_file(path_to_active)
    checks = read_csv_file(path_to_checks)
    
    # проверим, есть ли дополнительные данные.
    # если путь к файлу не пустой, то создаем датафрейм с допданными, соединяем его с основным датафреймом groups.
    if path_to_add != "":
        groups_add = read_csv_file(path_to_add)
        groups = pd.concat([groups, groups_add])
    
    # при необходимости выгружаем файл с данными groups:   
    if upload:
        groups.to_csv(path, sep=',', index=False, encoding='utf-8')
        
    # Присваиваем колонкам имена, используемые внутри функции, и приводим к нужным типам данные:
    groups.columns       = ['id', 'grp']
    active_studs.columns = ['id']
    checks.columns       = ['id', 'rev']
    
    groups       = groups.astype({'id':'int', 'grp':'str'})
    active_studs = active_studs.astype({'id':'int'})
    checks       = checks.astype({'id':'int', 'rev':'float'})
    
     # проверяем на наличие пропусков, используя функцию nan_checking:
    is_nan=[]
    is_nan.extend([
                nan_checking(groups),
                nan_checking(active_studs),
                nan_checking(checks)
                ])
    
    if all(is_nan):
        display(HTML('<font color="green"> Пропусков в данных нет </font>'))
    else:
        display(HTML('<font color="red", size=6> В каких-то таблицах есть пропуски! </font>'))
        print(is_nan)
    
    # проверяем колонки с id на уникальность, используя функцию is_unique:
    is_uniq=[]
    is_uniq.extend([
                is_unique(groups.id),
                is_unique(active_studs.id),
                is_unique(checks.id)
                ])

    if all(is_uniq):
        display(HTML('<font color="green"> Все значения id в таблицах уникальны </font>'))
    else:
        display(HTML('<font color="red", size=6> В каких-то таблицах есть повторяющиеся id! </font>'))
        print(is_uniq)
        
    # создаем единый датафрейм, в котором только активные пользователи (заходили в период эксперимента):     
    stud_info=groups\
        .merge(active_studs, how = 'inner', on = 'id')\
        .merge(checks, how = 'left', on = 'id')\
        .fillna(0) # заменим NaN значения на 0
    stud_info['rev'] = stud_info['rev'].round(2) # округлим значения выручки до 2 знаков после запятой
    
    # Возвращаем сформированный датафрейм:
    return stud_info
    

def plotter (
    data
    ):
    '''
        функция plotter выполняет построение графиков:
            - Соотношение общего количества активных пользователей и пользователей, совершивших покупки (countplot).
            - Распределение пользователей в группах по размеру оплат (гистограмма).
            - Характеристики распределения (boxplot). 
        ---
        Параметры:
            data : pandas.DataFrame 
                датафрейм, содержащий столбцы:  id пользователя / группа (А или В) / выручка.
        ---
        Результат: графики
    '''
    
    #-----подготовка данных-----
    
    # Присваиваем колонкам имена, используемые внутри функции, и приводим к нужным типам данные:
    data.columns = ['id', 'grp', 'rev']
    data = data.astype({'id':'int', 'grp':'str','rev':'float'})
    
    #создадим еще одну колонку с категориальным признаком оплаты:
    data['is_pay'] = data['rev'].apply(lambda x: 'купившие' if x > 0 else 'не купившие')
    
    #-----построение графиков-----
    
    #количество пользователей в контрольной и тестовой группах (с разделением купивших и не купивших)
    number_of_users = px.histogram(data, x='grp', 
                                   color='is_pay',
                                   labels={'grp':'группы'},
                                   title='Количество пользователей в группах (А - контроль, В - тест)')
    
    #остальные графики выведем по два с помощью subplots.
    
    # slot1 - строка из двух графиков : ARPU и ARPPU
    slot1_titles=['ARPU', 'ARPPU']
    arpu = go.Histogram(x=data.grp, y=data.rev, histfunc='avg')
    arppu = go.Histogram(x=data.query('rev>0').grp, y=data.query('rev>0').rev, histfunc='avg')

    slot1 = make_subplots(
        rows=1, cols=2,
        vertical_spacing=0.05,
        subplot_titles=slot1_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]
              ]
    )

    slot1.add_trace(arpu, row=1, col=1)
    slot1.add_trace(arppu, row=1, col=2)

    slot1.update_layout(
        height=500,
        showlegend=False,
        title_text="Средняя выручка (A - контроль, В - тест)",
        title_x=0.5
    )
    
    # slot2 - строка из двух графиков : Распределение выручки по посетителям
    slot2_titles=['контрольная группа', 'тестовая группа']
    dist_c = go.Histogram(x=data.query('grp=="A"').rev)
    dist_t = go.Histogram(x=data.query('grp=="B"').rev)


    slot2 = make_subplots(
        rows=1, cols=2,
        #shared_yaxes=True,
        vertical_spacing=0.05,
        subplot_titles=slot2_titles,
        specs=[[{"type": "histogram"}, {"type": "histogram"}]
              ]
    )

    slot2.add_trace(dist_c, row=1, col=1)
    slot2.add_trace(dist_t, row=1, col=2)

    slot2.update_layout(
        height=500,
        showlegend=False,
        title_text="Распределение выручки по посетителям сайта",
        title_x=0.5
    )
    
    # slot3 - строка из двух графиков : распределение выручки по покупателям
    slot3_titles=['контрольная группа', 'тестовая группа']
    dist_pay_c = go.Histogram(x=data.query('rev>0 and grp=="A"').rev)
    dist_pay_t = go.Histogram(x=data.query('rev>0 and grp=="B"').rev)

    slot3 = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        vertical_spacing=0.05,
        subplot_titles=slot3_titles,
        specs=[[{"type": "histogram"}, {"type": "histogram"}]
              ]
    )

    slot3.add_trace(dist_pay_c, row=1, col=1)
    slot3.add_trace(dist_pay_t, row=1, col=2)

    slot3.update_layout(
        height=500,
        showlegend=False,
        title_text="Распределение выручки по покупателям",
        title_x=0.5
    )
    
    # slot4 - строка из двух графиков : боксплоты
    slot4_titles=['все посетители', 'только купившие']

    boxtrace1 = go.Box(x=data.grp,  y=data.rev)
    boxtrace2 = go.Box(x=data.query('rev > 0').grp,  y=data.query('rev > 0').rev)


    slot4 = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        vertical_spacing=0.05,
        subplot_titles=slot4_titles,
        specs=[[{"type": "box"}, {"type": "box"}]
              ]
    )

    slot4.add_trace(boxtrace1, row=1, col=1)
    slot4.add_trace(boxtrace2, row=1, col=2)

    slot4.update_layout(
        height=500,
        showlegend=False,
        title_text="Диаграммы размаха boxplots (А - контроль, В - тест)",
        title_x=0.5
    )
    
    # выводим графики:
    
    number_of_users.show()
    slot1.show()
    slot2.show()
    slot3.show()
    slot4.show()

def metric_calculator (
    data,
    report=False
    ):
    '''
        функция metric_calculator выполняет расчет метрик для контрольной и тестовой групп:
            - Конверсия посетителей в покупку.
            - ARPU.
            - ARPPU. 
        ---
        Параметры:
            data    : pandas.DataFrame 
                датафрейм, содержащий столбцы:  id пользователя / группа (А или В) / выручка.
            report  : boolean, default False
                параметр определяет, нужно ли выводить отчет в консоль.
        ---
        Результат: metric_calculate_Result
            Кортеж, содержащий следующие параметры (c - для контрольной группы А, t - для тестовой группы B):
            cr=(cr_c, cr_t)
            arpu=(arpu_c, arpu_t)
            arppu=(arppu_c, arppu_t)
    '''
    
    #-----подготовка данных-----
    
    # Присваиваем колонкам имена, используемые внутри функции, и приводим к нужным типам данные:
    data.columns = ['id', 'grp', 'rev']
    data = data.astype({'id':'int', 'grp':'str','rev':'float'})
    
    # сформируем два  датафрейма для контрольной и тестовой групп:
    control = data.query('grp == "A"')
    test    = data.query('grp == "B"')
    
    #----- Расчет метрик -----
    
    # расчет CR:
    cr_c = control.rev[control.rev > 0].count() / control.id.count()
    cr_t = test.rev[test.rev > 0].count() / test.id.count()    
    # расчет ARPU:
    arpu_c = control.rev.sum() / control.id.count()
    arpu_t = test.rev.sum() / test.id.count()    
    # расчет ARPPU:
    arppu_c = control.rev.sum() / control.id[control.rev > 0].count()
    arppu_t = test.rev.sum() / test.id[test.rev > 0].count()
    
    #----- Отправка результатов -----
    
    # готовим именованный кортеж для вывода результата:
    resTuple=namedtuple('metrics_calculate_Result', 'cr arpu arppu')
    result=resTuple((cr_c, cr_t), (arpu_c, arpu_t), (arppu_c, arppu_t))
    
    # при необходимости выводим отчет:
    if report:
        print(f'''
                Количество пользователей, активных в период эксперимента:         {data.id.count()}
                Количество пользователей, выполнивших оплаты (из числа активных): {data.id[data.rev>0].count()}
                
                Конверсия в контрольной группе: {cr_c:.2%}
                Конверсия в тестовой группе:    {cr_t:.2%}
                ARPU контрольной группы: {arpu_c:.2f}
                ARPU тестовой группы:    {arpu_t:.2f}
                ARPPU контрольной группы: {arppu_c:.2f}
                ARPPU тестовой группы:    {arppu_t:.2f}
        ''')
        
    return result
    
def hyp_tester (
    data,
    metrics=['cr', 'arpu', 'arppu'],
    report=False
    ):
    '''
        функция hyp_tester выполняет расчет p-value выбранной метрики 
        (cr - методом bootstrap, arpu, arppu - t-тестом). 
        ---
        Параметры:
            data    : pandas.DataFrame 
                датафрейм, содержащий столбцы:  id пользователя / группа (А или В) / выручка.
            metrics : list, delaut ['cr', 'arpu', 'arppu']
                перечень метрик, которые нужно оценить.
            report  : boolean, default False
                параметр определяет, нужно ли выводить отчет в консоль.
        ---
        Результат: hyp_tester_Result
            Кортеж, содержащий следующие параметры (c - для контрольной группы А, t - для тестовой группы B):
            cr=(cr_c, cr_t)
            arpu=(arpu_c, arpu_t)
            arppu=(arppu_c, arppu_t)
    '''
    
    #-----подготовка данных-----
    
    # Присваиваем колонкам имена, используемые внутри функции, и приводим к нужным типам данные:
    data.columns = ['id', 'grp', 'rev']
    data = data.astype({'id':'int', 'grp':'str','rev':'float'})
    
    # создадим колонку is_pay:
    data['is_pay'] = data['rev'].apply(lambda x: 1 if x > 0 else 0)
    
    # сформируем два  датафрейма для контрольной и тестовой групп:
    control = data.query('grp == "A"')
    test    = data.query('grp == "B"')
    
    # создадим переменные для сохранения результатов расчета:
    cr_pv             = None
    arpu_c_intervals  = None
    arpu_t_intervals  = None
    arppu_c_intervals = None
    arppu_t_intervals = None
    
    # создадим переменные для формирования текстового отчета:
    
    rep_cr    = '\n            Тест на проверку конверсий не проводился \n'
    rep_arpu  = '\n            Тест на проверку ARPU не проводился \n'
    rep_arppu = '\n            Тест на проверку ARPPU не проводился \n'
    
    #-----выполняем расчеты-----
    
    # проверка различий в конверсии
    if 'cr' in metrics:
        cr_pv = sps.ttest_ind(control.is_pay, test.is_pay).pvalue
        if cr_pv < 0.05:
            rep_cr = f'''
            значение p-value {cr_pv} меньше 0.05, 
            с большой вероятностью различия в конверсии вызваны тестируемым нововведением.
            '''
        else:
            rep_cr = f'''
            значение p-value {cr_pv} больше 0.05, 
            это не позволяет подтвердить гипотезу о статистической значимости отличий конверсии
            в контрольной и тестовой группах.
            '''
    
    # проверка различий в ARPU:
    if 'arpu' in metrics:
        arpu_c_intervals = sps.bootstrap((control.rev,), np.mean).confidence_interval
        arpu_t_intervals = sps.bootstrap((test.rev,), np.mean).confidence_interval
        #определяем, пересекаются ли доверительные интервалы метрики в группах
        low_c  = arpu_c_intervals.low        
        high_c = arpu_c_intervals.high
        low_t  = arpu_t_intervals.low
        high_t = arpu_t_intervals.high

        if (high_c > low_t and high_c < high_t) or (low_c < high_t and low_c > low_t):
            rep_arpu = f'''
            bootstrap-тест ARPU показал:
            Доверительные интервалы выборок средних пересекаются.
            Нулевая гипотеза не может быть отвергнута.
            '''
        else:
            rep_arpu = f'''
            bootstrap-тест ARPU показал:
            Доверительные интервалы выборок средних не пересекаются.
            С большой вероятностью различия в ARPU вызваны тестируемым нововведением
            '''
        
    # проверка различий в ARPPU:
    if 'arppu' in metrics:
        arppu_c_intervals = sps.bootstrap((control.query('rev > 0').rev,), np.mean).confidence_interval
        arppu_t_intervals = sps.bootstrap((test.query('rev > 0').rev,), np.mean).confidence_interval
        #определяем, пересекаются ли доверительные интервалы метрики в группах
        low_c = arppu_c_intervals.low        
        high_c = arppu_c_intervals.high
        low_t = arppu_t_intervals.low
        high_t = arppu_t_intervals.high

        if (high_c > low_t and high_c < high_t) or (low_c < high_t and low_c > low_t):
            rep_arppu = f'''
            bootstrap-тест ARPPU показал:
            Доверительные интервалы выборок средних пересекаются.
            Нулевая гипотеза не может быть отвергнута.
            '''
        else:
            rep_arppu = f'''
            bootstrap-тест ARPPU показал:
            Доверительные интервалы выборок средних не пересекаются.
            С большой вероятностью различия в ARPPU вызваны тестируемым нововведением
            '''
        
    #----- Подготовка результатов -----
    
    # готовим именованный кортеж для вывода результата:
    resTuple=namedtuple('hyp_tester_Result', 'pvalue_of_cr arpu_intervals arppu_intervals')
    result=resTuple((cr_pv), (arpu_c_intervals, arpu_t_intervals), (arppu_c_intervals, arppu_t_intervals))
        
    if report:
        print(rep_cr + rep_arpu + rep_arppu)
        
    return result