# -*- coding: utf-8 -*-
"""
author: linchen
time: 2018/12/4 上午12:25
"""

import os
import numpy as np
import pandas as pd
import datetime
from math import radians, atan, tan, sin, acos, cos
from sklearn.cluster import DBSCAN


def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter

    rb = 6356755  # radius of polar: meter

    flatten = (ra - rb) / ra  # Partial rate of the earth

    # change angle to radians

    radLatA = radians(latA)

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

        return distance
    except:

        return 0.0000001


def half_hour(t):
    if t <= 30:
        return (0)
    else:
        return (0.5)


def getplacelist(df, n=10):
    start_counts = df['start_place'].value_counts()
    start_place_list = pd.DataFrame(start_counts.index, columns=['place'])
    start_counts.index = start_place_list.index
    start_place_list['counts_x'] = start_counts

    end_counts = df['end_place'].value_counts()
    end_place_list = pd.DataFrame(end_counts.index, columns=['place'])
    end_counts.index = end_place_list.index
    end_place_list['counts_y'] = end_counts

    place_list = pd.merge(start_place_list, end_place_list, on=['place', 'place'], how='outer')
    place_list = place_list.fillna(value=0)
    place_list['counts_a'] = place_list['counts_x'] + place_list['counts_y']
    place_list_awlays = place_list.sort_values(by=['counts_y', 'counts_a', 'counts_x'], axis=0, ascending=False).head(n)
    place_list_awlays.index = np.arange(start=0, stop=place_list_awlays.shape[0], step=1)
    return place_list_awlays


def getpredictendplace(out_id, start_place, start_half_hour, week_day,
                       whether_holiday, whether_firstdayof_holiday,
                       whether_endsdayof_holiday, daynum_of_holiday,
                       whether_weekend, whether_workday,
                       train_flag, data):
    # 不是冷启动，只考虑该out_id的数据作为参考
    train_data = train_flag[train_flag['out_id'] == out_id]
    target_data = data[data['out_id'] == out_id]

    # 考虑P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]取其最大值
    # P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend)]
    # = P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # *p[target]
    # /P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]
    # p[target]已知
    # 计算P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # target_data['p[conditions|target']
    # 遇到没有数据可以参考，减少条件
    conditions_data = train_data[
        (train_data['start_place'] == start_place) &
        (train_data['start_half_hour'] == start_half_hour) &
        (train_data['week_day'] == week_day) &
        (train_data['whether_holiday'] == whether_holiday) &
        (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
        (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday) &
        (train_data['daynum_of_holiday'] == daynum_of_holiday) &
        (train_data['whether_weekend'] == whether_weekend) &
        (train_data['whether_workday'] == whether_workday)
        ]
    conditions_counts = conditions_data['r_key'].count()
    if conditions_counts == 0:
        # 去掉week_day
        conditions_data = train_data.loc[
            (train_data['start_place'] == start_place) &
            (train_data['start_half_hour'] == start_half_hour) &
            (train_data['whether_holiday'] == whether_holiday) &
            (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
            (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday) &
            (train_data['daynum_of_holiday'] == daynum_of_holiday) &
            (train_data['whether_weekend'] == whether_weekend) &
            (train_data['whether_workday'] == whether_workday)
            ]
        conditions_counts = conditions_data['r_key'].count()
        if conditions_counts == 1:
            return conditions_data.iat[0, 30]
        if (conditions_counts == 0) & (whether_holiday == 1):
            # 去掉daynum_of_holiday
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                (train_data['start_half_hour'] == start_half_hour) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
                (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday)
                ]
            conditions_counts = conditions_data.count()
            if conditions_counts == 1:
                return conditions_data.iat[0, 30]
            if conditions_counts == 0:
                # 去掉是否第一天、是否最后一天
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    (train_data['start_half_hour'] == start_half_hour) &
                    (train_data['whether_holiday'] == whether_holiday)]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts == 1:
                    return conditions_data.iat[0, 30]
                if conditions_counts == 0:
                    # 如果是6-23点，时间范围扩展2小时
                    if (start_half_hour < 23) and (start_half_hour > 5):
                        conditions_data = train_data.loc[
                            (train_data['start_place'] == start_place) &
                            ((train_data['start_half_hour'] <= start_half_hour + 2) &
                             (train_data['start_half_hour'] >= start_half_hour - 2)
                             ) &
                            (train_data['whether_holiday'] == whether_holiday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 1:
                            return conditions_data.iat[0, 30]
                        if conditions_counts == 0:
                            # 去掉出发地点
                            conditions_data = train_data.loc[
                                ((train_data['start_half_hour'] <= start_half_hour + 2) &
                                 (train_data['start_half_hour'] >= start_half_hour - 2)
                                 ) &
                                (train_data['whether_holiday'] == whether_holiday)
                                ]
                            conditions_counts = conditions_data['r_key'].count()
                            if conditions_counts == 1:
                                return conditions_data.iat[0, 30]
                            if conditions_counts == 0:
                                return -1
                            else:
                                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        # 23-5点，看整个时段的数据
                        conditions_data = train_data.loc[
                            (train_data['start_place'] == start_place) &
                            ((train_data['start_half_hour'] >= 23) |
                             (train_data['start_half_hour'] <= 5)
                             ) &
                            (train_data['whether_holiday'] == whether_holiday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 1:
                            return conditions_data.iat[0, 30]
                        if conditions_counts == 0:
                            # 去掉出发地点
                            conditions_data = train_data.loc[
                                ((train_data['start_half_hour'] >= 23) |
                                 (train_data['start_half_hour'] <= 5)
                                 ) &
                                (train_data['whether_holiday'] == whether_holiday)
                                ]
                            conditions_counts = conditions_data['r_key'].count()
                            if conditions_counts == 1:
                                return conditions_data.iat[0, 30]
                            if conditions_counts == 0:
                                return -1
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        if (conditions_counts == 0) & (whether_workday == 1):
            if (start_half_hour < 23) and (start_half_hour > 5):
                # 放宽时间范围，前后2小时
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] <= start_half_hour + 2) &
                     (train_data['start_half_hour'] >= start_half_hour - 2)
                     ) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts == 1:
                    return conditions_data.iat[0, 30]
                if conditions_counts == 0:
                    # 放宽出发地点
                    conditions_data = train_data.loc[
                        ((train_data['start_half_hour'] <= start_half_hour + 2) &
                         (train_data['start_half_hour'] >= start_half_hour - 2)
                         ) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts == 1:
                        return conditions_data.iat[0, 30]
                    if conditions_counts == 0:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                # 放宽到23-5点
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] >= 23) |
                     (train_data['start_half_hour'] <= 5)
                     ) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts == 1:
                    return conditions_data.iat[0, 30]
                if conditions_counts == 0:
                    # 放宽目的地
                    conditions_data = train_data.loc[
                        ((train_data['start_half_hour'] >= 23) |
                         (train_data['start_half_hour'] <= 5)
                         ) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts == 1:
                        return conditions_data.iat[0, 30]
                    if conditions_counts == 0:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        if (conditions_counts == 0) & (whether_weekend == 1):
            if (start_half_hour < 23) and (start_half_hour > 5):
                # 放宽时间限制，2小时
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] <= start_half_hour + 2) &
                     (train_data['start_half_hour'] >= start_half_hour - 2)
                     ) &
                    (train_data['whether_weekend'] == whether_weekend)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts == 1:
                    return conditions_data.iat[0, 30]
                if conditions_counts == 0:
                    # 放宽出发地点
                    conditions_data = train_data.loc[

                        ((train_data['start_half_hour'] <= start_half_hour + 2) &
                         (train_data['start_half_hour'] >= start_half_hour - 2)
                         ) &
                        (train_data['whether_weekend'] == whether_weekend)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts == 1:
                        return conditions_data.iat[0, 30]
                    if conditions_counts == 0:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                # 放宽时间5-23点
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] >= 23) |
                     (train_data['start_half_hour'] <= 5)
                     ) &
                    (train_data['whether_weekend'] == whether_weekend)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts == 1:
                    return conditions_data.iat[0, 30]
                if conditions_counts == 0:
                    # 放宽出发地点
                    conditions_data = train_data.loc[
                        ((train_data['start_half_hour'] >= 23) |
                         (train_data['start_half_hour'] <= 5)
                         ) &
                        (train_data['whether_weekend'] == whether_weekend)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts == 1:
                        return conditions_data.iat[0, 30]
                    if conditions_counts == 0:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        else:
            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    else:
        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    tmp = pd.DataFrame()
    tmp['conditions_counts_by_target'] = conditions_counts_by_target
    tmp['target'] = tmp.index
    result = target_data.merge(tmp, on='target', how='inner')
    result['p[conditions|target]'] = result['conditions_counts_by_target'] / result['target_counts']
    result['P[target|conditions]'] = result['P[target]'] * result['p[conditions|target]']
    return result.sort_values(by='P[target|conditions]', axis=0, ascending=False).iat[0, 3]


def getpredictendplacev2(out_id, start_place, start_half_hour, week_day,
                         whether_holiday, whether_firstdayof_holiday,
                         whether_endsdayof_holiday, daynum_of_holiday,
                         whether_weekend, whether_workday,
                         train_flag, data):
    # 不是冷启动，只考虑该out_id的数据作为参考
    train_data = train_flag[train_flag['out_id'] == out_id]
    target_data = data[data['out_id'] == out_id]

    # 考虑P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]取其最大值
    # P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend)]
    # = P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # *p[target]
    # /P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]
    # p[target]已知
    # 计算P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # target_data['p[conditions|target']
    # 遇到没有数据可以参考，减少条件
    conditions_data = train_data[
        (train_data['start_place'] == start_place) &
        (train_data['start_half_hour'] == start_half_hour) &
        (train_data['whether_holiday'] == whether_holiday) &
        (train_data['whether_weekend'] == whether_weekend) &
        (train_data['whether_workday'] == whether_workday)
        ]
    conditions_counts = conditions_data['r_key'].count()
    if conditions_counts <= 3:
        # 放宽时间限制
        if (start_half_hour > 5) and (start_half_hour < 23):
            begin_time = start_half_hour - 2
            if start_half_hour + 2 > 24:
                end_time = start_half_hour - 22
            else:
                end_time = start_half_hour + 2
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                (train_data['start_half_hour'] >= begin_time) &
                (train_data['start_half_hour'] <= end_time) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_weekend'] == whether_weekend) &
                (train_data['whether_workday'] == whether_workday)
                ]
            conditions_counts = conditions_data['r_key'].count()
            if conditions_counts <= 3:
                conditions_data = train_data.loc[
                    (train_data['start_half_hour'] == start_half_hour) &

                    (train_data['whether_holiday'] == whether_holiday) &
                    (train_data['whether_weekend'] == whether_weekend) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 3:
                    conditions_data = train_data.loc[

                        (train_data['start_half_hour'] >= begin_time) &
                        (train_data['start_half_hour'] <= end_time) &
                        (train_data['whether_holiday'] == whether_holiday) &
                        (train_data['whether_weekend'] == whether_weekend) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <= 3:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        else:
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                ((train_data['start_half_hour'] >= 23) |
                 (train_data['start_half_hour'] <= 5)) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_weekend'] == whether_weekend) &
                (train_data['whether_workday'] == whether_workday)
                ]
            conditions_counts = conditions_data['r_key'].count()
            if conditions_counts <= 3:
                conditions_data = train_data.loc[
                    (train_data['start_half_hour'] == start_half_hour) &
                    (train_data['whether_holiday'] == whether_holiday) &
                    (train_data['whether_weekend'] == whether_weekend) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 3:
                    conditions_data = train_data.loc[
                        ((train_data['start_half_hour'] >= 23) |
                         (train_data['start_half_hour'] <= 5)) &
                        (train_data['whether_holiday'] == whether_holiday) &
                        (train_data['whether_weekend'] == whether_weekend) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <= 3:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    else:
        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    tmp = pd.DataFrame()
    tmp['conditions_counts_by_target'] = conditions_counts_by_target
    tmp['target'] = tmp.index
    result = target_data.merge(tmp, on='target', how='inner')
    result['p[conditions|target]'] = result['conditions_counts_by_target'] / result['target_counts']
    result['P[target|conditions]'] = result['P[target]'] * result['p[conditions|target]']
    return result.sort_values(by='P[target|conditions]', axis=0, ascending=False).iat[0, 3]


def getpredictendplacev3(out_id, start_place, start_half_hour, week_day,
                         whether_holiday, whether_firstdayof_holiday,
                         whether_endsdayof_holiday, daynum_of_holiday,
                         whether_weekend, whether_workday,
                         train_flag, data):
    # 不是冷启动，只考虑该out_id的数据作为参考
    train_data = train_flag[train_flag['out_id'] == out_id]
    target_data = data[data['out_id'] == out_id]

    # 考虑P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]取其最大值
    # P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend)]
    # = P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # *p[target]
    # /P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]
    # p[target]已知
    # 计算P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # target_data['p[conditions|target']
    # 遇到没有数据可以参考，减少条件
    conditions_data = train_data[
        (train_data['start_place'] == start_place) &
        (train_data['start_half_hour'] == start_half_hour) &
        (train_data['week_day'] == week_day) &
        (train_data['whether_holiday'] == whether_holiday) &
        (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
        (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday) &
        (train_data['daynum_of_holiday'] == daynum_of_holiday) &
        (train_data['whether_weekend'] == whether_weekend) &
        (train_data['whether_workday'] == whether_workday)
        ]
    conditions_counts = conditions_data['r_key'].count()
    if conditions_counts <= 1:
        # 去掉week_day
        conditions_data = train_data.loc[
            (train_data['start_place'] == start_place) &
            (train_data['start_half_hour'] == start_half_hour) &
            (train_data['whether_holiday'] == whether_holiday) &
            (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
            (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday) &
            (train_data['daynum_of_holiday'] == daynum_of_holiday) &
            (train_data['whether_weekend'] == whether_weekend) &
            (train_data['whether_workday'] == whether_workday)
            ]
        conditions_counts = conditions_data['r_key'].count()
        if (conditions_counts <= 5) & (whether_holiday == 1):
            # 去掉daynum_of_holiday
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                (train_data['start_half_hour'] == start_half_hour) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_firstdayof_holiday'] == whether_firstdayof_holiday) &
                (train_data['whether_endsdayof_holiday'] == whether_endsdayof_holiday)
                ]
            conditions_counts = conditions_data['r_key'].count()
            if conditions_counts <= 5:
                # 去掉是否第一天、是否最后一天
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    (train_data['start_half_hour'] == start_half_hour) &
                    (train_data['whether_holiday'] == whether_holiday)]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 5:
                    # 如果是6-23点，时间范围扩展2小时
                    if (start_half_hour < 23) & (start_half_hour > 5):
                        start_time = start_half_hour - 2
                        if start_half_hour + 2 >= 24:
                            end_time = start_half_hour - 22
                        else:
                            end_time = start_half_hour + 2
                        conditions_data = train_data.loc[
                            (train_data['start_place'] == start_place) &
                            ((train_data['start_half_hour'] <= end_time) &
                             (train_data['start_half_hour'] >= start_time)
                             ) &
                            (train_data['whether_holiday'] == whether_holiday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts <= 5:
                            # 去掉出发地点
                            conditions_data = train_data.loc[
                                (train_data['start_half_hour'] == start_half_hour)
                                &
                                (train_data['whether_holiday'] == whether_holiday)
                                ]
                            conditions_counts = conditions_data['r_key'].count()
                            if conditions_counts <= 5:
                                conditions_data = train_data.loc[

                                    ((train_data['start_half_hour'] <= end_time) &
                                     (train_data['start_half_hour'] >= start_time)
                                     ) &
                                    (train_data['whether_holiday'] == whether_holiday)
                                    ]
                                conditions_counts = conditions_data['r_key'].count()
                                if conditions_counts == 0:
                                    return -1
                                else:
                                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                            else:
                                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        # 23-5点，看整个时段的数据
                        conditions_data = train_data.loc[
                            (train_data['start_place'] == start_place) &
                            ((train_data['start_half_hour'] >= 23) |
                             (train_data['start_half_hour'] <= 5)
                             ) &
                            (train_data['whether_holiday'] == whether_holiday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts <= 5:
                            # 去掉出发地点
                            conditions_data = train_data.loc[
                                (train_data['start_half_hour'] == start_half_hour)

                                &
                                (train_data['whether_holiday'] == whether_holiday)
                                ]
                            conditions_counts = conditions_data['r_key'].count()
                            if conditions_counts <= 5:
                                conditions_data = train_data.loc[

                                    ((train_data['start_half_hour'] >= 23) |
                                     (train_data['start_half_hour'] <= 5)
                                     ) &
                                    (train_data['whether_holiday'] == whether_holiday)
                                    ]
                                conditions_counts = conditions_data['r_key'].count()
                                if conditions_counts == 0:
                                    return -1
                                else:
                                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        if (conditions_counts <= 1) & (whether_workday == 1):
            if (start_half_hour < 23) & (start_half_hour > 5):
                # 放宽时间范围，前后2小时
                start_time = start_half_hour - 2
                if start_half_hour + 2 >= 24:
                    end_time = start_half_hour - 22
                else:
                    end_time = start_half_hour + 2
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] <= end_time) &
                     (train_data['start_half_hour'] >= start_time)
                     ) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()

                if conditions_counts <= 5:
                    # 放宽出发地点
                    conditions_data = train_data.loc[
                        (train_data['start_half_hour'] == start_half_hour) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <= 5:
                        conditions_data = train_data.loc[

                            ((train_data['start_half_hour'] <= end_time) &
                             (train_data['start_half_hour'] >= start_time)
                             ) &
                            (train_data['whether_workday'] == whether_workday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 0:
                            return -1
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                # 放宽到23-5点
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] >= 23) |
                     (train_data['start_half_hour'] <= 5)
                     ) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 5:
                    # 放宽目的地
                    conditions_data = train_data.loc[
                        (train_data['start_half_hour'] == start_half_hour) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <= 5:
                        conditions_data = train_data.loc[

                            ((train_data['start_half_hour'] >= 23) |
                             (train_data['start_half_hour'] <= 5)
                             ) &
                            (train_data['whether_workday'] == whether_workday)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 0:
                            return -1
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        if (conditions_counts == 0) & (whether_weekend == 1):
            if (start_half_hour < 23) & (start_half_hour > 5):
                start_time = start_half_hour - 2
                if start_half_hour + 2 >= 24:
                    end_time = start_half_hour - 22
                else:
                    end_time = start_half_hour + 2
                # 放宽时间限制，2小时
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] <= end_time) &
                     (train_data['start_half_hour'] >= start_time)
                     ) &
                    (train_data['whether_weekend'] == whether_weekend)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 5:
                    # 放宽出发地点
                    conditions_data = train_data.loc[

                        (train_data['start_half_hour'] == start_half_hour)
                        &
                        (train_data['whether_weekend'] == whether_weekend)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <= 5:
                        conditions_data = train_data.loc[

                            ((train_data['start_half_hour'] <= end_time) &
                             (train_data['start_half_hour'] >= start_time)
                             ) &
                            (train_data['whether_weekend'] == whether_weekend)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 0:
                            return -1
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                # 放宽时间5-23点
                conditions_data = train_data.loc[
                    (train_data['start_place'] == start_place) &
                    ((train_data['start_half_hour'] >= 23) |
                     (train_data['start_half_hour'] <= 5)
                     ) &
                    (train_data['whether_weekend'] == whether_weekend)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 5:
                    # 放宽出发地点
                    conditions_data = train_data.loc[
                        (train_data['start_half_hour'] == start_half_hour) &
                        (train_data['whether_weekend'] == whether_weekend)
                        ]
                    conditions_counts = conditions_data['r_key'].count()

                    if conditions_counts <= 5:
                        conditions_data = train_data.loc[

                            ((train_data['start_half_hour'] >= 23) |
                             (train_data['start_half_hour'] <= 5)
                             ) &
                            (train_data['whether_weekend'] == whether_weekend)
                            ]
                        conditions_counts = conditions_data['r_key'].count()
                        if conditions_counts == 0:
                            return -1
                        else:
                            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        else:
            conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    else:
        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    tmp = pd.DataFrame()
    tmp['conditions_counts_by_target'] = conditions_counts_by_target
    tmp['target'] = tmp.index
    result = target_data.merge(tmp, on='target', how='inner')
    result['p[conditions|target]'] = result['conditions_counts_by_target'] / result['target_counts']
    result['P[target|conditions]'] = result['P[target]'] * result['p[conditions|target]']
    return result.sort_values(by='P[target|conditions]', axis=0, ascending=False).iat[0, 3]


'''
def getpredictendplacev2(out_id, start_place, start_half_hour, week_day,
                         whether_holiday, whether_firstdayof_holiday,
                         whether_endsdayof_holiday, daynum_of_holiday,
                         whether_weekend, whether_workday,
                         train_flag, data):
    # 不是冷启动，只考虑该out_id的数据作为参考
    train_data = train_flag[train_flag['out_id'] == out_id]
    target_data = data[data['out_id'] == out_id]

    # 考虑P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]取其最大值
    # P[target|(start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend)]
    # = P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # *p[target]
    # /P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend]
    # p[target]已知
    # 计算P[start_place,start_half_hour,whether_holiday,whether_firstdayof_holiday,whether_endsdayof_holiday,daynum_of_holiday,whether_workday,weekday,whether_weekend|target]
    # target_data['p[conditions|target']
    # 遇到没有数据可以参考，减少条件
    conditions_data = train_data[
        (train_data['start_place'] == start_place) &
        (train_data['start_half_hour'] == start_half_hour) &
        (train_data['whether_holiday'] == whether_holiday) &
        (train_data['whether_weekend'] == whether_weekend) &
        (train_data['whether_workday'] == whether_workday)
        ]
    conditions_counts = conditions_data['r_key'].count()
    if conditions_counts <= 3:
        # 放宽时间限制
        if (start_half_hour > 5) and (start_half_hour < 23):
            begin_time = start_half_hour-2
            if start_half_hour+2>24:
                end_time = start_half_hour-22
            else:
                end_time = start_half_hour+2
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                (train_data['start_half_hour'] >= begin_time) &
                (train_data['start_half_hour'] <= end_time) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_weekend'] == whether_weekend) &
                (train_data['whether_workday'] == whether_workday)
                ]
            conditions_counts = conditions_data['r_key'].count()
            if conditions_counts <=3:
                conditions_data = train_data.loc[
                    (train_data['start_half_hour'] == start_half_hour) &

                    (train_data['whether_holiday'] == whether_holiday) &
                    (train_data['whether_weekend'] == whether_weekend) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <=3:
                    conditions_data = train_data.loc[

                        (train_data['start_half_hour'] >= begin_time) &
                        (train_data['start_half_hour'] <= end_time) &
                        (train_data['whether_holiday'] == whether_holiday) &
                        (train_data['whether_weekend'] == whether_weekend) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts<=3:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
        else:
            conditions_data = train_data.loc[
                (train_data['start_place'] == start_place) &
                ((train_data['start_half_hour'] >= 23) |
                 (train_data['start_half_hour'] <= 5)) &
                (train_data['whether_holiday'] == whether_holiday) &
                (train_data['whether_weekend'] == whether_weekend) &
                (train_data['whether_workday'] == whether_workday)
                ]
            conditions_counts = conditions_data['r_key'].count()
            if conditions_counts <= 3:
                conditions_data = train_data.loc[
                    (train_data['start_half_hour'] == start_half_hour) &
                    (train_data['whether_holiday'] == whether_holiday) &
                    (train_data['whether_weekend'] == whether_weekend) &
                    (train_data['whether_workday'] == whether_workday)
                    ]
                conditions_counts = conditions_data['r_key'].count()
                if conditions_counts <= 3:
                    conditions_data = train_data.loc[
                        ((train_data['start_half_hour'] >= 23) |
                         (train_data['start_half_hour'] <= 5)) &
                        (train_data['whether_holiday'] == whether_holiday) &
                        (train_data['whether_weekend'] == whether_weekend) &
                        (train_data['whether_workday'] == whether_workday)
                        ]
                    conditions_counts = conditions_data['r_key'].count()
                    if conditions_counts <=3:
                        return -1
                    else:
                        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
                else:
                    conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
            else:
                conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    else:
        conditions_counts_by_target = conditions_data.groupby('target')['r_key'].count()
    tmp = pd.DataFrame()
    tmp['conditions_counts_by_target'] = conditions_counts_by_target
    tmp['target'] = tmp.index
    result = target_data.merge(tmp, on='target', how='inner')
    result['p[conditions|target]'] = result['conditions_counts_by_target'] / result['target_counts']
    result['P[target|conditions]'] = result['P[target]'] * result['p[conditions|target]']
    return result.sort_values(by='P[target|conditions]', axis=0, ascending=False).iat[0, 3]
'''


def getnum(t, n=0):
    tmp = t.split("_")[n]
    tmp = tmp[0:len(tmp) - 3] + "." + tmp[-3:]
    return float(tmp)


def get3num(t):
    num_x, num_y = str(t).split('.')
    num = num_x + num_y[0:3]
    return num


if __name__ == '__main__':
    # 修改工作目录
    os.chdir('/Users/linchen/IdeaProjects/machine-learning/src/main/python/dcjingsai/car-destination')

    # 获取数据
    print('获取数据')
    test_new = pd.read_csv("data/test_new.csv", low_memory=False)
    train_new = pd.read_csv("data/train_new.csv", low_memory=False)
    data_date = pd.read_csv("data/data_date.csv", low_memory=False)
    data_date['start_time_datetime'] = data_date['start_time'].map(pd.to_datetime)
    data_start_date = data_date['start_time_datetime'].dt.date
    data_date['start_date'] = data_start_date

    # 时间特征处理####################################################################
    # 转换时间类型
    print('转换时间类型')
    start_time_datetime = train_new["start_time"].map(pd.to_datetime)
    train_new['start_time_datetime'] = start_time_datetime
    end_time_datetime = train_new["end_time"].map(pd.to_datetime)
    train_new['end_time_datetime'] = end_time_datetime

    start_time_datetime = test_new["start_time"].map(pd.to_datetime)
    test_new['start_time_datetime'] = start_time_datetime

    # 每30分钟进行划分-特征half_hour
    start_hour = train_new["start_time_datetime"].map(lambda t: t.hour)
    # half_hour = train_new_test["start_time_datetime"].map(lambda t: t.minute).map(half_hour)
    half_hour = train_new["start_time_datetime"].apply(lambda row: 0 if row.minute < 30 else 0.5)
    start_half_hour = start_hour + half_hour
    train_new.insert(train_new.shape[1], "start_hour", start_hour)  # insert无返回值
    train_new.insert(train_new.shape[1], "start_half_hour", start_half_hour)

    start_hour = test_new["start_time_datetime"].map(lambda t: t.hour)
    # half_hour = train_new_test["start_time_datetime"].map(lambda t: t.minute).map(half_hour)
    half_hour = test_new["start_time_datetime"].apply(lambda row: 0 if row.minute < 30 else 0.5)
    start_half_hour = start_hour + half_hour
    test_new.insert(test_new.shape[1], "start_hour", start_hour)  # insert无返回值
    test_new.insert(test_new.shape[1], "start_half_hour", start_half_hour)

    # 取出发日期-特征start_date
    start_date = train_new['start_time_datetime'].dt.date
    train_new['start_date'] = start_date

    start_date = test_new['start_time_datetime'].dt.date
    test_new['start_date'] = start_date

    # 合并节假日等数据-特征
    # weekday 周几
    # whether_holiday 是否节假日
    # daynum_of_holiday 节假日第几天
    # whether_firstdayof_holiday 是否节假日第一天
    # whether_endsdayof_holiday  是否节假日最后一天
    # whether_workday 是否工作日
    # whether_weekend 是否周末
    print('合并data_date')
    train_new = pd.merge(train_new, data_date, left_on='start_date', right_on='start_date', how='left')
    test_new = pd.merge(test_new, data_date, left_on='start_date', right_on='start_date', how='left')

    # 地理特征处理####################################################################
    # 经纬度信息处理-出发地点，保留3位小数后合并经纬度

    print('地理特征处理')
    start_lon_round = train_new['start_lon'].map(lambda t: get3num(t))
    start_lat_round = train_new['start_lat'].map(lambda t: get3num(t))
    start_place = start_lon_round + "_" + start_lat_round
    train_new['start_place'] = start_place

    start_lon_round = test_new['start_lon'].map(lambda t: get3num(t))
    start_lat_round = test_new['start_lat'].map(lambda t: get3num(t))
    start_place = start_lon_round + "_" + start_lat_round
    test_new['start_place'] = start_place

    # 经纬度信息处理-目的地，保留3位小数后合并经纬度
    end_lon_round = train_new['end_lon'].map(lambda t: get3num(t))
    end_lat_round = train_new['end_lat'].map(lambda t: get3num(t))
    end_place = end_lon_round + "_" + end_lat_round
    train_new['end_place'] = end_place

    train = train_new.copy()
    test = test_new.copy()

    # 划分训练集-测试集，训练集1-7月15日不含端午节数据，7月28-8月1日数据,2018-06-16 2018-06-17 2018-06-18
    """train = train_new[train_new.start_time_datetime_x.dt.month <= 5]
    train = train.append(train_new[(train_new.start_time_datetime_x.dt.month == 6)
                                   & (train_new.start_time_datetime_x.dt.day != 16)
                                   & (train_new.start_time_datetime_x.dt.day != 17)
                                   & (train_new.start_time_datetime_x.dt.day != 18)
                                   ])
    train = train.append(train_new[(train_new.start_time_datetime_x.dt.month == 7)
                                   & (train_new.start_time_datetime_x.dt.day <= 27)])

    test = train_new[(train_new.start_time_datetime_x.dt.month == 7)
                     & (train_new.start_time_datetime_x.dt.day >= 28)]

    test = test.append(train_new[(train_new.start_time_datetime_x.dt.month == 6)
                                 & ((train_new.start_time_datetime_x.dt.day == 16)
                                    | (train_new.start_time_datetime_x.dt.day == 17)
                                    | (train_new.start_time_datetime_x.dt.day == 18)
                                    )
                                 ]
                       )

    train.shape
    test.shape"""
    # 1-7月中上旬最常去的10个地方，作为备选点，部分样本备选点不足10个，flag为0-9，其他目的地为-1
    print("获取第一阶段备选数据")
    placelist = train.groupby(['out_id']).apply(getplacelist)
    placelist['out_id'] = placelist.index.get_level_values(0)
    placelist['place_id'] = placelist.index.get_level_values(1)

    train_flag = pd.merge(train, placelist, left_on=['out_id', 'end_place'],
                          right_on=['out_id', 'place'], how='left')
    train_flag['target'] = train_flag.apply(lambda row: -1 if np.isnan(row.place_id) else row.place_id, axis=1)

    '''test_flag = pd.merge(test, placelist, left_on=['out_id', 'end_place'],
                         right_on=['out_id', 'place'], how='left')
    test_flag['target'] = test_flag.apply(lambda row: -1 if np.isnan(row.place_id) else row.place_id, axis=1)
    '''
    # 第一步筛选出out_id对应数据
    # 第二步朴素贝叶斯，考虑
    # start_place
    # half_hour
    #  weekday 周几
    #  whether_holiday 是否节假日:
    #  whether_firstdayof_holiday 是否节假日第一天
    #  whether_endsdayof_holiday  是否节假日最后一天
    #  daynum_of_holiday 节假日第几天
    #  whether_weekend 是否周末
    #  whether_workday 是否工作日

    # 按out_id汇总数据
    groupby = train_flag.groupby('out_id')
    data = pd.DataFrame()

    # 总次数
    pname = 'sample_counts'
    tmp = groupby['r_key'].count()
    data[pname] = tmp
    data['out_id'] = tmp.index

    # 计算每个人目的地出现的概率
    tmp = pd.DataFrame()
    pname = 'target_counts'
    tmp[pname] = groupby['target'].value_counts()
    tmp['out_id'] = tmp.index.get_level_values(0)
    tmp['target'] = tmp.index.get_level_values(1)
    data = data.merge(tmp, on='out_id', how='left')
    data['P[target]'] = data['target_counts'] / data['sample_counts']

    """
    Probability = {}
    smooth_y = 10.
    smooth_x = 0.

    name = 'start_block'
    pname = 'P(start_block|end_block)'
    print('calculating %s' % pname)
    dy = train_flag.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = train_flag.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]
    """


    #  输入
    print('第一阶段预测')
    predict_place_id = []
    id = []
    r = []
    for i in range(test_new.shape[0]):
        # i=16
        r_key = test_new.iat[i, 0]
        out_id = test_new.iat[i, 1]
        start_place = test_new.iat[i, 19]
        start_half_hour = test_new.iat[i, 7]
        week_day = test_new.iat[i, 11]
        whether_holiday = test_new.iat[i, 12]
        whether_firstdayof_holiday = test_new.iat[i, 14]
        whether_endsdayof_holiday = test_new.iat[i, 15]
        daynum_of_holiday = test_new.iat[i, 13]
        whether_weekend = test_new.iat[i, 17]
        whether_workday = test_new.iat[i, 16]
        tmp = getpredictendplacev3(out_id=out_id, start_place=start_place, start_half_hour=start_half_hour,
                                   week_day=week_day,
                                   whether_holiday=whether_holiday,
                                   whether_firstdayof_holiday=whether_firstdayof_holiday,
                                   whether_endsdayof_holiday=whether_endsdayof_holiday,
                                   daynum_of_holiday=daynum_of_holiday,
                                   whether_weekend=whether_weekend, whether_workday=whether_workday,
                                   train_flag=train_flag, data=data)
        # print(tmp)
        predict_place_id.append(tmp)
        id.append(out_id)
        r.append(r_key)
        print(i)


    # start
    result = pd.DataFrame({'r_key': r,
                           'out_id': id,
                           'predict_place_id': predict_place_id
                           })
    result = result.merge(placelist, left_on=['out_id', 'predict_place_id'], right_on=['out_id', 'place_id'],
                          how='left')
    predict_result_1 = result[result.predict_place_id != -1]

    e_lon = predict_result_1['place'].apply(lambda row: getnum(row))
    e_lat = predict_result_1['place'].apply(lambda row: getnum(row, n=1))
    predict_result_1['end_lon'] = e_lon
    predict_result_1['end_lat'] = e_lat
    predict_result_1.loc[:, ['r_key', 'end_lat', 'end_lon']].to_csv('data/result1.csv', index=None)

    # 进入second_stage_dbscan+bayes，尝试考虑out_id和不考虑out_id两种情况
    # second_stage如果不考虑out_id，是否可以使用树模型

    # ------ 1. 密度聚类: 对经纬度坐标点进行密度聚类
    trL = train_flag.shape[0] * 2  # 训练集行数*2: 合并训练集出发地与目的地经纬度
    # 合并训练集、测试集经纬度
    X = np.concatenate([train_flag[['start_lat', 'start_lon']].values,  # 训练集出发地经纬度
                        train_flag[['end_lat', 'end_lon']].values,  # 训练集目的地经纬度
                        test_new[['start_lat', 'start_lon']].values])

    # 密度聚类调参
    # https://www.cnblogs.com/pinard/p/6217852.html
    db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
    # plot_cluster(X[trL:, 0], X[trL:, 1], db.labels_)  # 画出测试集聚类结果
    labels = db.labels_  # 获取聚类结果label: array([   -1,     0,    -1, ..., 96102, 19658, 20044], dtype=int64)

    n_clusters_ = len(set(labels))  # 107466
    print('Estimated number of clusters: %d' % n_clusters_)  # 打印聚类数: 110227

    # -- 聚类结果分析
    # 训练集聚类结果分析
    info = pd.DataFrame(X[:trL, :], columns=['lat', 'lon'])
    info['block_id'] = labels[:trL]  # 训练集聚类label
    clear_info = info.loc[info.block_id != -1, :]
    print('The number of miss start block in train data', (info.block_id.iloc[:trL // 2] == -1).sum())  # 265579
    print('The number of miss end block in train data', (info.block_id.iloc[trL // 2:] == -1).sum())  # 268794
    # -- 测试集聚类结果分析
    test_info = pd.DataFrame(X[trL:, :], columns=['lat', 'lon'])
    test_info['block_id'] = labels[trL:]  # 测试集聚类label
    print('The number of miss start block in test data', (test_info.block_id == -1).sum())

    # ------ 2. 构造聚类label: 将聚类label拼接到训练集和测试集上
    train_flag['start_block'] = info.block_id.iloc[:trL // 2].values  # 将出发地聚类结果拼接到训练集
    train_flag['end_block'] = info.block_id.iloc[trL // 2:].values  # 将目的地聚类结果拼接到训练集
    test_new['start_block'] = test_info.block_id.values  # 将出发地聚类结果拼接到测试集
    # 聚类结果清洗: 去掉训练集中聚类标签为-1的样本
    good_train_idx = (train_flag.start_block != -1) & (train_flag.end_block != -1)
    print('The number of good training data', good_train_idx.sum())  # 保留样本数: 997161; 占比69%
    good_train = train_flag.loc[good_train_idx, :]

    tmp = test_new.copy()
    test_flag_s2 = tmp[result['predict_place_id'] == -1]

    end_block_counts = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(
        columns={'r_key': 'end_block_counts'})
    counts = good_train['r_key'].count()

    Probability = {}
    smooth_y = 10.
    smooth_x = 0.

    name = 'start_block'
    pname = 'P(start_block|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'out_id'
    pname = 'P(out_id|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'start_hour'
    pname = 'P(start_hour|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'week_day'
    pname = 'P(week_day|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'whether_workday'
    pname = 'P(whether_workday|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'whether_weekend'
    pname = 'P(whether_weekend|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'whether_holiday'
    pname = 'P(whether_holiday|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'whether_firstdayof_holiday'
    pname = 'P(whether_firstdayof_holiday|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'whether_endsdayof_holiday'
    pname = 'P(whether_endsdayof_holiday|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    name = 'daynum_of_holiday'
    pname = 'P(daynum_of_holiday|end_block)'
    print('calculating %s' % pname)
    dy = good_train.groupby('end_block', as_index=False)['r_key'].count().rename(columns={'r_key': 'y'})
    dx = good_train.groupby(['end_block', name], as_index=False)['r_key'].count().rename(columns={'r_key': 'x'})
    dxy = dx.merge(dy, on='end_block', how='left')
    dxy[pname] = (dxy.x + smooth_x) / (dxy.y.astype(float) + smooth_y)
    Probability[pname] = dxy[['end_block', name, pname]]

    pname = 'P(end_block)'
    print('calculating %s' % pname)
    tmp = good_train.end_block.value_counts().reset_index()  # [问题] 训练集目的地label个数统计，并非概率
    tmp.columns = ['end_block', pname]
    Probability[pname] = tmp

    predict_info = test_flag_s2.copy()  # predict_info 为测试集

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(out_id|end_block)'], on='out_id', how='left')
    print(predict_info['P(out_id|end_block)'].isnull().sum())  # join后无空值
    predict_info['P(out_id|end_block)'] = predict_info['P(out_id|end_block)'].fillna(1e-5)

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(start_block|end_block)'], on=['start_block', "end_block"],
                                      how='left')
    print(predict_info['P(start_block|end_block)'].isnull().sum())  # join后空值个数: 1017007
    predict_info['P(start_block|end_block)'] = predict_info['P(start_block|end_block)'].fillna(1e-5)

    # -- 条件概率

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(start_hour|end_block)'],
                                      on=['start_hour', 'end_block'], how='left')
    print(predict_info['P(start_hour|end_block)'].isnull().sum())  # join后空值个数: 897409
    predict_info['P(start_hour|end_block)'] = predict_info['P(start_hour|end_block)'].fillna(1e-4)

    predict_info = predict_info.merge(Probability['P(week_day|end_block)'], on=['week_day', 'end_block'], how='left')
    print(predict_info['P(week_day|end_block)'].min(),
          predict_info['P(week_day|end_block)'].isnull().sum())  # join后空值个数: 732126
    predict_info['P(week_day|end_block)'] = predict_info['P(week_day|end_block)'].fillna(1e-4)

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(whether_holiday|end_block)'], on=['whether_holiday', 'end_block'],
                                      how='left')
    print(predict_info['P(whether_holiday|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(whether_holiday|end_block)'] = predict_info['P(whether_holiday|end_block)'].fillna(1e-4)
    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(whether_workday|end_block)'], on=['whether_workday', 'end_block'],
                                      how='left')
    print(predict_info['P(whether_workday|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(whether_workday|end_block)'] = predict_info['P(whether_workday|end_block)'].fillna(1e-4)

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(whether_weekend|end_block)'], on=['whether_weekend', 'end_block'],
                                      how='left')
    print(predict_info['P(whether_weekend|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(whether_weekend|end_block)'] = predict_info['P(whether_weekend|end_block)'].fillna(1e-4)

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(whether_firstdayof_holiday|end_block)'],
                                      on=['whether_firstdayof_holiday', 'end_block'],
                                      how='left')
    print(predict_info['P(whether_firstdayof_holiday|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(whether_firstdayof_holiday|end_block)'] = predict_info[
        'P(whether_firstdayof_holiday|end_block)'].fillna(1e-4)

    # -- 条件概率
    predict_info = predict_info.merge(Probability['P(whether_endsdayof_holiday|end_block)'],
                                      on=['whether_endsdayof_holiday', 'end_block'],
                                      how='left')
    print(predict_info['P(whether_endsdayof_holiday|end_block)'].isnull().sum())  # join后空值个数: 223746
    predict_info['P(whether_endsdayof_holiday|end_block)'] = predict_info[
        'P(whether_endsdayof_holiday|end_block)'].fillna(1e-4)

    # -- 先验概率(训练集目的地聚类label先验概率)
    predict_info = predict_info.merge(Probability['P(end_block)'], on='end_block', how='left')
    print(predict_info['P(end_block)'].isnull().sum())  # join后无空值
    predict_info['P(end_block)'] = predict_info['P(end_block)'].fillna(1e-1)

    # ------ 7. 计算后验概率: 根据训练集得到的先验概率，计算测试集上目的地label的后验概率
    predict_info['P(end_block|(start_block, out_id, is_holiday, hour))'] = predict_info['end_block'] * \
                                                                           predict_info[
                                                                               'P(whether_firstdayof_holiday|end_block)'] * \
                                                                           predict_info[
                                                                               'P(whether_endsdayof_holiday|end_block)'] * \
                                                                           predict_info[
                                                                               'P(whether_weekend|end_block)'] * \
                                                                           predict_info[
                                                                               'P(whether_holiday|end_block)'] * \
                                                                           predict_info[
                                                                               'P(whether_workday|end_block)'] * \
                                                                           predict_info['P(week_day|end_block)'] * \
                                                                           predict_info['P(start_hour|end_block)'] * \
                                                                           predict_info['P(out_id|end_block)'] * \
                                                                           predict_info['P(start_block|end_block)']

    which_probability = 'P(end_block|(start_block, out_id, is_holiday, hour))'

    # ------ 8. 计算目的地聚类label对应的经纬度: 根据训练集生成每个聚类label的经纬度
    block_lat_lon = good_train.groupby('end_block')[['end_lat', 'end_lon']].mean().reset_index()  # 取每个聚类label下目的地经纬度的均值
    predict_info = predict_info.merge(block_lat_lon, on='end_block', how='left')  # 将目的地经纬度拼接到测试集
    print(predict_info[['start_lat', 'start_lon', 'end_lat', 'end_lon']].describe())

    # ------ 9. 获取预测结果: 将后验概率最大的目的地作为预测结果
    # pby('r_keypredict_result = predict_info.grou').apply(lambda g: g.loc[g[which_probability].idxmax(), :]).reset_index(drop=True)
    predict_result = predict_info.sort_values(by=[which_probability], ascending=False).groupby('r_key',
                                                                                               as_index=False).first()
    predict_result[['r_key', 'end_lat', 'end_lon']].to_csv('data/result2.csv', index=None)  # 保存预测结果

    # ------ tmp: 结果合并
    res1 = pd.read_csv("data/result1.csv", low_memory=False)
    res2 = pd.read_csv("data/result2.csv", low_memory=False)
    res = res1.append(res2)
    res.to_csv('data/result_lc.csv', index=None)  # 保存预测结果
