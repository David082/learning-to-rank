# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/28
version :
refer :
-- XGBoost多分类
https://blog.csdn.net/u010159842/article/details/53411355

2018-12-03
1. 修改先验概率: 0.42*****************5 [精度不变]
2.

"""
import pandas as pd
import numpy as np
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def plot_cluster(lat, lon, label):
    """画出聚类结果图
    """
    import matplotlib.pyplot as plt
    plt.scatter(lat, lon, c=label)
    plt.show()


def distance(latA, lonA, latB, lonB):
    ra = 6378140  # 赤道半径: 米
    rb = 6356755  # 极线半径: 米

    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)  # 弧度
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)

    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))

        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2

        dr = flatten / 8 * (c1 - c2)

        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


def f(d):
    return 1 / (1 + np.exp(-(d - 1000) / 250))


def transformer(df):
    """生成is_holiday和hour字段
    """
    special_holiday = ['2018-01-01'] + ['2018-02-%d' % d for d in range(15, 22)] + \
                      ['2018-04-%2d' % d for d in range(5, 8)] + \
                      ['2018-04-%d' % d for d in range(29, 31)] + ['2018-05-01'] + \
                      ['2018-06-%d' % d for d in range(16, 19)] + \
                      ['2018-09-%d' % d for d in range(22, 25)] + \
                      ['2018-10-%2d' % d for d in range(1, 8)]
    special_workday = ['2018-02-%d' % d for d in [11, 24]] + \
                      ['2018-04-08'] + ['2018-04-28'] + \
                      ['2018-09-%d' % d for d in range(29, 31)]
    for t_col in ['start_time']:
        tmp = df[t_col].map(pd.Timestamp)
        df['hour'] = tmp.map(lambda t: t.hour // 3)
        df['half'] = tmp.map(lambda t: t.minute // 30)
        df['day'] = tmp.map(lambda t: t.dayofweek)
        tmp_date = df[t_col].map(lambda s: s.split(' ')[0])
        not_spworkday_idx = ~tmp_date.isin(special_workday)
        spholiday_idx = tmp_date.isin(special_holiday)
        weekend_idx = (df['day'] >= 5)
        df['is_holiday'] = ((weekend_idx & not_spworkday_idx) | spholiday_idx).astype(int)


if __name__ == '__main__':
    train = pd.read_csv('data/train_new.csv', low_memory=False)  # 训练集行数: 1495814
    test = pd.read_csv('data/test_new.csv', low_memory=False)  # 测试集行数: 58097

    # ------ 1. 密度聚类: 对经纬度坐标点进行密度聚类
    trL = train.shape[0] * 2  # 训练集行数*2: 合并训练集出发地与目的地经纬度
    # 合并训练集、测试集经纬度
    X = np.concatenate([train[['start_lat', 'start_lon']].values,  # 训练集出发地经纬度
                        train[['end_lat', 'end_lon']].values,  # 训练集目的地经纬度
                        test[['start_lat', 'start_lon']].values])

    # 密度聚类调参
    # https://www.cnblogs.com/pinard/p/6217852.html
    db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
    # plot_cluster(X[trL:, 0], X[trL:, 1], db.labels_)  # 画出测试集聚类结果
    labels = db.labels_  # 获取聚类结果label: array([   -1,     0,    -1, ..., 96102, 19658, 20044], dtype=int64)

    n_clusters_ = len(set(labels))
    print('Estimated number of clusters: %d' % n_clusters_)  # 打印聚类数: 110227

    # -- 聚类结果分析
    # 训练集聚类结果分析
    info = pd.DataFrame(X[:trL, :], columns=['lat', 'lon'])
    info['block_id'] = labels[:trL]  # 训练集聚类label
    clear_info = info.loc[info.block_id != -1, :]
    print('The number of miss start block in train data', (info.block_id.iloc[:trL // 2] == -1).sum())  # 271285
    print('The number of miss end block in train data', (info.block_id.iloc[trL // 2:] == -1).sum())  # 274566
    # -- 测试集聚类结果分析
    test_info = pd.DataFrame(X[trL:, :], columns=['lat', 'lon'])
    test_info['block_id'] = labels[trL:]  # 测试集聚类label
    print('The number of miss start block in test data', (test_info.block_id == -1).sum())

    # ------ 2. 构造聚类label: 将聚类label拼接到训练集和测试集上
    train['start_block'] = info.block_id.iloc[:trL // 2].values  # 将出发地聚类结果拼接到训练集
    train['end_block'] = info.block_id.iloc[trL // 2:].values  # 将目的地聚类结果拼接到训练集
    test['start_block'] = test_info.block_id.values  # 将出发地聚类结果拼接到测试集
    # 聚类结果清洗: 去掉训练集中聚类标签为-1的样本
    good_train_idx = (train.start_block != -1) & (train.end_block != -1)
    print('The number of good training data', good_train_idx.sum())  # 保留样本数: 1033722; 占比69%
    good_train = train.loc[good_train_idx, :]
    print('saving new train & test data')
    good_train.to_csv('data/good_train.csv', index=None)
    test.to_csv('data/good_test.csv', index=None)  # 测试集原样保留

    # ------ 3. 特征构造: 为训练集和测试集生成is_holiday和hour字段
    train = pd.read_csv('data/good_train.csv', low_memory=False)  # 重新读取训练集: 样本减少
    test = pd.read_csv('data/good_test.csv', low_memory=False)  # 重新读取测试集
    transformer(train)  # 生成特征字段
    transformer(test)  # 生成特征字段

    # ------ 4. 计算条件概率: 根据训练集, 计算朴素贝叶斯算法需要使用的条件概率(基于目的地label的条件概率)
    Probability = {}
    smooth_y = 10.
    smooth_x = 0.
    ## P(start_block|end_block)
    name = 'start_block'
    pname = 'P(start_block|end_block)'
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]
    ## P(out_id|end_block)
    name = 'out_id'
    pname = 'P(out_id|end_block)'
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]
    ## P(is_holiday|end_block)
    name = 'is_holiday'
    pname = 'P(is_holiday|end_block)'
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]
    ## P((is_holiday, hour)|end_block)
    pname = 'P((is_holiday, hour)|end_block)'
    name = ['is_holiday', 'hour']
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block'] + name, as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', pname] + name]
    ## P(day|end_block)
    name = 'day'
    pname = 'P(day|end_block)'
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]
    ## P(hour|end_block)
    name = 'hour'
    pname = 'P(hour|end_block)'
    print('calculating %s' % pname)
    tmp_func = lambda g: 1.0 * g[name].value_counts() / len(g)
    tmp = train.groupby('end_block').apply(tmp_func).reset_index()
    tmp.columns = ['end_block', name, pname]
    Probability[pname] = tmp
    ## P((hour,half)|end_block)
    pname = 'P((hour,half)|end_block)'
    print('calculating %s' % pname)
    dy = train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    # ------ 5. 计算先验概率: 根据训练集计算目的地label先验概率
    pname = 'P(end_block)'
    print('calculating %s' % pname)
    tmp = train.end_block.value_counts().reset_index()  # [问题] 训练集目的地label个数统计，并非概率
    tmp.columns = ['end_block', pname]
    tmp[pname] = tmp[pname].apply(lambda x: x / tmp[pname].sum())  # ------ add
    Probability[pname] = tmp

    # ------ 6. 概率拼接: 将根据训练集计算出的条件概率、先验概率拼接到测试集
    # P(end_block|(start_block, out_id, is_holiday, hour)) =
    # P(end_block) * P(start_block|end_block) * P(out_id|end_block) * P((is_holiday, hour)|end_block)
    is_local = False  # 是否线下验证
    if is_local:
        predict_info = train.copy()
        predict_info = predict_info.rename(
            columns={'end_block': 'true_end_block', 'end_lat': 'true_end_lat', 'end_lon': 'true_end_lon'}
        )
    else:
        predict_info = test.copy()  # predict_info 为测试集
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(out_id|end_block)'], on='out_id', how='left')
    print(predict_info['P(out_id|end_block)'].isnull().sum())  # join后无空值
    predict_info['P(out_id|end_block)'] = predict_info['P(out_id|end_block)'].fillna(1e-5)
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(is_holiday|end_block)'], on=['is_holiday', 'end_block'],
                                      how='left')
    print(predict_info['P(is_holiday|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(is_holiday|end_block)'] = predict_info['P(is_holiday|end_block)'].fillna(1e-4)
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(day|end_block)'], on=['day', 'end_block'], how='left')
    print(predict_info['P(day|end_block)'].min(), predict_info['P(day|end_block)'].isnull().sum())  # join后空值个数: 732126
    predict_info['P(day|end_block)'] = predict_info['P(day|end_block)'].fillna(1e-4)
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P((is_holiday, hour)|end_block)'],
                                      on=['is_holiday', 'hour', 'end_block'], how='left')
    print(predict_info['P((is_holiday, hour)|end_block)'].isnull().sum())  # join后空值个数: 897409
    predict_info['P((is_holiday, hour)|end_block)'] = predict_info['P((is_holiday, hour)|end_block)'].fillna(1e-4)
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(start_block|end_block)'], on=['start_block', 'end_block'],
                                      how='left')
    print(predict_info['P(start_block|end_block)'].isnull().sum())  # join后空值个数: 1017007
    predict_info['P(start_block|end_block)'] = predict_info['P(start_block|end_block)'].fillna(1e-5)
    # -- 先验概率(训练集目的地聚类label先验概率)
    predict_info = predict_info.merge(Probability['P(end_block)'], on='end_block', how='left')
    print(predict_info['P(end_block)'].isnull().sum())  # join后无空值
    predict_info['P(end_block)'] = predict_info['P(end_block)'].fillna(1e-1)

    # ------ 7. 计算后验概率: 根据训练集得到的先验概率，计算测试集上目的地label的后验概率
    predict_info['P(end_block|(start_block, out_id, is_holiday, hour))'] = predict_info[
                                                                               'P((is_holiday, hour)|end_block)'] * \
                                                                           predict_info['P(out_id|end_block)'] * \
                                                                           predict_info['P(start_block|end_block)'] * \
                                                                           predict_info['P(end_block)']
    which_probability = 'P(end_block|(start_block, out_id, is_holiday, hour))'

    # ------ 8. 计算目的地聚类label对应的经纬度: 根据训练集生成每个聚类label的经纬度
    block_lat_lon = train.groupby('end_block')[['end_lat', 'end_lon']].mean().reset_index()  # 取每个聚类label下目的地经纬度的均值
    predict_info = predict_info.merge(block_lat_lon, on='end_block', how='left')  # 将目的地经纬度拼接到测试集
    print(predict_info[['start_lat', 'start_lon', 'end_lat', 'end_lon']].describe())

    """模型融合
    """

    # ------ 9. 获取预测结果: 将后验概率最大的目的地作为预测结果
    # predict_result = predict_info.groupby('r_key').apply(lambda g: g.loc[g[which_probability].idxmax(), :]).reset_index(drop=True)
    predict_result = predict_info.sort_values(by=[which_probability], ascending=False).groupby('r_key',
                                                                                               as_index=False).first()
    predict_result[['r_key', 'end_lat', 'end_lon']].to_csv('data/result.csv', index=None)  # 保存预测结果

    # ------ 10. 冷启动问题
    # 目前测试集暂无冷启动问题
    if not is_local:
        output_result = test[['r_key', 'start_lat', 'start_lon']].merge(predict_result[['r_key', 'end_lat', 'end_lon']],
                                                                        on='r_key', how='left')
        print(output_result.end_lat.isnull().sum())
        # 冷启动暂时用出发地经纬度作为预测结果
        nan_idx = output_result.end_lat.isnull()
        output_result.loc[nan_idx, 'end_lat'] = output_result['start_lat'][nan_idx]
        output_result.loc[nan_idx, 'end_lon'] = output_result['start_lon'][nan_idx]
        # output_result[['start_lat', 'end_lat', 'end_lon']].describe()
        print(output_result.head())
        print(output_result.info())
        output_result[['r_key', 'end_lat', 'end_lon']].to_csv('data/result.csv', index=None)  # 保存预测结果
