import json

import ollama
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from nutri_details.EER import pal_info, get_eer_age_range
from nutri_details.Fat import get_protein_fat_age_range, carbohydrates_info

from nutri_details.protein import ear_rni_info

def apipei(query, choices):
    match_scores = [(choice, fuzz.ratio(query, choice)) for choice in choices]
    match_scores.sort(key=lambda x: x[1], reverse=True)
    return match_scores[0][0], match_scores[0][1]


def bpipei(query, choices):
    scorer = fuzz.QRatio
    result = process.extractOne(query, choices, scorer=scorer)
    return result[0] if result else None


nutrient_dataset_path = 'nutrient_dataset.txt'
df_nutrient = pd.read_csv(nutrient_dataset_path, encoding='utf-8')
glycemic_index_dataset_path = 'food_GI.csv'
df_glycemic_index = pd.read_csv(glycemic_index_dataset_path)
insulin_index_dataset_path = 'Insulin_index.csv'
df_insulin_index = pd.read_csv(insulin_index_dataset_path)
df_glycemic_index = df_glycemic_index.dropna()
df_glycemic_index['Name'] = df_glycemic_index['Name'].str.lower()


def match_info_from_db(food_name):
    matched_food1, match_score1 = apipei(food_name, df_nutrient['Descrip'])
    matched_food2 = bpipei(food_name, df_glycemic_index['Name'])
    matched_food3 = bpipei(food_name, df_insulin_index['Name'])
    print(matched_food1)
    filtered_df1 = df_nutrient.loc[
        df_nutrient['Descrip'] == matched_food1, ['能量_千卡', '蛋白质_克', '脂肪_克', '碳水化合物_克', '糖_克',
                                                  '纤维_克', '维生素A_微克', '维生素B6_毫克', '维生素B12_微克',
                                                  '维生素C_毫克', '维生素E_毫克', '叶酸_微克', '烟酸_毫克',
                                                  '核黄素_毫克', '硫胺素_毫克', '钙_毫克', '铜_毫克', '铁_毫克',
                                                  '镁_毫克', '锰_毫克', '磷_毫克', '硒_微克', '锌_毫克',
                                                  '维生素A_推荐量', '维生素B6_推荐量', '维生素B12_推荐量',
                                                  '维生素C_推荐量', '维生素E_推荐量', '叶酸_推荐量', '烟酸_推荐量',
                                                  '核黄素_推荐量', '硫胺素_推荐量', '钙_推荐量', '铜_推荐量',
                                                  '镁_推荐量', '磷_推荐量', '硒_推荐量', '锌_推荐量']]
    print(filtered_df1)
    filtered_df2 = df_glycemic_index[df_glycemic_index['Name'] == matched_food2]
    filtered_df3 = df_insulin_index[df_insulin_index['Name'] == matched_food3]
    return filtered_df1, filtered_df2, filtered_df3


def parse_nutri_info(nutri_info):
    rows = []
    for item in nutri_info.keys():
        row_content = []
        if not "推荐量" in item:
            nutrient = item.split("_")[0]  # 获取营养成分名
            unit = item.split("_")[1]
            current_value = nutri_info[item]  # to ceil
            if f"{nutrient}_推荐量" in nutri_info:
                recommended_value = nutri_info[f"{nutrient}_推荐量"]  # to ceil
                recommended_value = round(recommended_value, 4)
                recommended_value = "{:.2f}%".format(recommended_value * 100)
            else:
                recommended_value = "无数据"
            row_content.append(nutrient)
            row_content.append(f"{current_value}({unit})")
            row_content.append(recommended_value)
    print(f"row{rows}")
    result_dict = {
        "name": "能量供给",
        "score": 3.5,
        "scoreText": "这个食品非常适合你吃",
        "RowLabels": ["项目", "当前值（单位）", "百克占比"],
        "RowContents": rows,
    }
    return result_dict


def parse_sugar_info(nutri_info, gi_info_list, ii_info_list):
    gi_info_row = ["GI(升糖指数)", gi_info_list[0]]
    ii_info_row = ["II(胰岛素指数)", ii_info_list[0]]
    carb_info_row = ["碳水化合物(克)", nutri_info["碳水化合物_克"]]
    sugar_info_row = ["糖(克)", nutri_info["糖_克"]]
    fiber_info_row = ["纤维(克)", nutri_info["纤维_克"]]

    result_dict = {
        "name": "糖代谢",
        "score": 4.5,
        "scoreLabel": "适合",
        "scoreText": "这个食品糖:",
        "RowLabels": ["项目", "值"],
        "RowContents": [gi_info_row, ii_info_row, carb_info_row, sugar_info_row, fiber_info_row]
    }
    return result_dict


def parse_eer_info(eer_dict):
    pal_mj_info_row = ["能量需要量（兆焦耳）", eer_dict["PAL(MJ)"]]
    pal_kcal_info_row = ["能量需要量（千卡路里）", eer_dict["PAL(kcal)"]]

    result_dict = {
        "name": "膳食能量需要量信息",
        "score": 4.5,
        "scoreLabel": "适合",
        "scoreText": "此食品的EER：满足机体总能量消耗所需的能量",
        "RowLabels": ["项目", "值"],
        "RowContents": [pal_mj_info_row, pal_kcal_info_row]
    }
    return result_dict


def parse_protein_info(protein_dict):
    ear_info_row = ["平均需要量", protein_dict["EAR"]]
    rni_info_row = ["能量需要量（千卡路里）", protein_dict["RNI"]]

    result_dict = {
        "name": "蛋白质推荐量信息",
        "score": 4.5,
        "scoreLabel": "适合",
        "scoreText": "平均需要量：群体中各个体营养素需要量的平均值。\n 推荐摄入量：可以满足某一特定性别、年龄及生理状况群体中绝大多数个体需要的营养素摄入水平。",
        "RowLabels": ["项目", "值"],
        "RowContents": [ear_info_row, rni_info_row]
    }
    return result_dict


def parse_fat_carb_info(fat_carb_dict):
    age_row = ["年龄(岁)/生理状况", fat_carb_dict["年龄(岁)/生理状况"]]
    carb_ear_row = ["碳水化合物平均需要量", fat_carb_dict["碳水化合物EAR"]]
    carb_amdr_row = ["碳水化合物每日摄入量范围", fat_carb_dict["碳水化合物AMDR"]]
    n6_unsat_fat_ai = ["n-6多不饱和脂肪酸\n安全摄入水平", fat_carb_dict["n-6多不饱和脂肪酸AI"]]
    n6_unsat_fat_amdr = ["n-6多不饱和脂肪酸\n每日摄入量范围", fat_carb_dict["n-6多不饱和脂肪酸AMDR"]]
    n3_unsat_fat_ai = ["n-3多不饱和脂肪酸\n安全摄入水平", fat_carb_dict["n-3多不饱和脂肪酸AIᵇ"]]
    n3_unsat_fat_amdr = ["n-3多不饱和脂肪酸\n每日摄入量范围", fat_carb_dict["n-3多不饱和脂肪酸AMDR"]]

    result_dict = {
        "name": "碳水、脂肪推荐量信息",
        "score": 4.5,
        "scoreLabel": "适合",
        "scoreText": "平均需要量：群体中各个体营养素需要量的平均值。\n 推荐摄入量：可以满足某一特定性别、年龄及生理状况群体中绝大多数个体需要的营养素摄入水平。",
        "RowLabels": ["项目", "值"],
        "RowContents": [age_row, carb_ear_row, carb_amdr_row, n6_unsat_fat_ai, n6_unsat_fat_amdr, n3_unsat_fat_ai,
                        n3_unsat_fat_amdr]
    }
    return result_dict


def gen_result_detail(nutri_info_list, gi_info_list, ii_info_list, eer_info_list, protein_info_list,
                      fat_carb_info_list):
    result_detail = [parse_nutri_info(nutri_info_list), parse_sugar_info(nutri_info_list, gi_info_list, ii_info_list),
                     parse_eer_info(eer_info_list), parse_protein_info(protein_info_list),
                     parse_fat_carb_info(fat_carb_info_list)]
    return result_detail


def ask_llm(food_name, df1, df2, df3, user_desc):
    nutri_info = df1.to_json(orient='records')
    GI_info = df2.to_json(orient='records')
    II_info = df3.to_json(orient='records')
    message = f"{food_name}（营养素信息：{nutri_info}，升糖指数：{GI_info}，胰岛素指数：{II_info}输出结果不需要显示）适合{user_desc}的人吗？（只需要给出合理的建议）"
    response = ollama.chat(model='llama2-chinese', messages=[
        {
            'role': 'user',
            'content': message,
        },
    ])
    return response


def handle_food_info_get(food_name, user_desc, user_info):
    filtered_df1, filtered_df2, filtered_df3 = match_info_from_db(food_name)
    # user_info_parse
    user_age = user_info["age"]
    user_gender = user_info["gender"]
    user_PA = user_info["PA"]

    # Nutri_Info GI II
    nutri_info_list = filtered_df1.to_dict(orient='records')
    GI_info_list = filtered_df2['Glycemic index'].tolist()
    II_info_list = filtered_df3['Insulin index'].tolist()

    # EER
    age = float(user_age)
    age_range = get_eer_age_range(age)
    PAL_MJ, PAL_kcal = pal_info(age_range, user_gender, user_PA)
    eer_info_list = {
        "PAL(MJ)": PAL_MJ,
        "PAL(kcal)": PAL_kcal
    }

    # PROTEIN
    age_range = get_protein_fat_age_range(age)
    EAR, RNI = ear_rni_info(age_range, user_gender)
    protein_info_list = {
        "EAR": EAR,
        "RNI": RNI
    }

    # FAT_CARB
    age_range = get_protein_fat_age_range(age)
    fat_carb_info_list = carbohydrates_info(age_range)

    user_desc=gen_user_desc(user_info)
    # ai_response = ask_llm(food_name, filtered_df1, filtered_df2, filtered_df3, user_desc)
    with open("./trial-reply.txt", "r", encoding="utf-8") as file:
        content = file.read()
    output = {
        "result_detail": gen_result_detail(nutri_info_list[0], GI_info_list, II_info_list, eer_info_list,
                                           protein_info_list, fat_carb_info_list),
        # "ai_response": response['message']['content']
        "ai_response": content,

    }
    print(json.dumps(output, indent=4, ensure_ascii=False))
    return output

def gen_user_desc(user_info):
    labels=user_info["userLabelData"]
    age=user_info["age"]
    gender=user_info["gender"]
    PA=user_info["PA"]

    return f"年龄：{age}，性别：{gender}，运动量：{PA}，身体情况：{'、'.join(labels)}"

# handle_food_info_get("apple", "24岁健康男性", user_info)

