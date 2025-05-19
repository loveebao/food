import pandas as pd
import json

data =  pd.read_csv(r"./nutri_details/中国居民脂肪碳水推荐量.csv")

def get_protein_fat_age_range(age):
    if age < 0.5:
        return '0~'
    elif age < 1:
        return '0.5~'
    elif age < 4:
        return '1~'
    elif age < 7:
        return '4~'
    elif age < 11:
        return '7~'
    elif age < 14:
        return '11~'
    elif age < 18:
        return '14~'
    elif age < 60:
        return '18~'
    else:
        return '60~'
    
def carbohydrates_info(age_range):
    matched_rows = data[data['年龄(岁)/生理状况'].str.contains(age_range)]
    if len(matched_rows) == 0:
        return "输入的年龄不在数据集中,请检查。"
    else:
        row = matched_rows.iloc[0]  
        return row.to_dict()


