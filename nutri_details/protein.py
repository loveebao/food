import pandas as pd
import json

from nutri_details.Fat import get_protein_fat_age_range

data = pd.read_csv(r"./nutri_details/中国居民蛋白质推荐量.csv")





def ear_rni_info(age_range, gender):
    try:
        matched_rows = data[data['年龄(岁)/生理状况'].str.contains(age_range)]
        if len(matched_rows) == 0:
            raise ValueError("输入的年龄不在数据集中，请检查。")
        else:
            if gender == "男":
                EAR = matched_rows.iloc[0, 1]
                RNI = matched_rows.iloc[0, 2]
            elif gender == "女":
                EAR = matched_rows.iloc[0, 3]
                RNI = matched_rows.iloc[0, 4]
            else:
                raise ValueError("输入的性别有误，请检查。")
        return EAR, RNI
    except ValueError as e:
        print(e)
    except Exception as e:
        print("发生了一个错误：", e)






