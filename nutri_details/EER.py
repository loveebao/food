import pandas as pd
import json

data = pd.read_csv(r"./nutri_details/中国居民膳食能量需要量(EER).csv")


def get_eer_age_range(age):
    if age < 0.5:
        return '0~'
    elif age < 1:
        return '0.5~'
    elif age < 2:
        return '1~'
    elif age < 3:
        return '2~'
    elif age < 4:
        return '3~'
    elif age < 5:
        return '4~'
    elif age < 6:
        return '5~'
    elif age < 7:
        return '6~'
    elif age < 8:
        return '7~'
    elif age < 9:
        return '8~'
    elif age < 10:
        return '9~'
    elif age < 11:
        return '10~'
    elif age < 14:
        return '11~'
    elif age < 18:
        return '14~'
    elif age < 50:
        return '18~'
    elif age < 65:
        return '50~'
    elif age < 80:
        return '65~'
    else:
        return '80~'


def pal_info(age_range, gender, PA):
    try:
        clean_data = data.dropna(subset=['年龄(岁)/生理状况'])
        matched_rows = clean_data[clean_data['年龄(岁)/生理状况'].str.contains(age_range)]
        if len(matched_rows) == 0:
            raise ValueError("输入的年龄不在数据集中，请检查。")
        else:
            if gender == "男":
                if PA == "轻":
                    PAL_MJ = matched_rows.iloc[0, 1]
                    PAL_kcal = matched_rows.iloc[0, 2]
                elif PA == "中":
                    PAL_MJ = matched_rows.iloc[0, 3]
                    PAL_kcal = matched_rows.iloc[0, 4]
                elif PA == "重":
                    PAL_MJ = matched_rows.iloc[0, 5]
                    PAL_kcal = matched_rows.iloc[0, 6]
                else:
                    raise ValueError("输入的PA有误，请检查。")
            elif gender == "女":
                if PA == "轻":
                    PAL_MJ = matched_rows.iloc[0, 7]
                    PAL_kcal = matched_rows.iloc[0, 8]
                elif PA == "中":
                    PAL_MJ = matched_rows.iloc[0, 9]
                    PAL_kcal = matched_rows.iloc[0, 10]
                elif PA == "重":
                    PAL_MJ = matched_rows.iloc[0, 11]
                    PAL_kcal = matched_rows.iloc[0, 12]
                else:
                    raise ValueError("输入的PA有误，请检查。")
            else:
                raise ValueError("输入的性别有误，请检查。")
        return PAL_MJ, PAL_kcal
    except ValueError as e:
        print(e)
    except Exception as e:
        print("发生了一个错误：", e)

