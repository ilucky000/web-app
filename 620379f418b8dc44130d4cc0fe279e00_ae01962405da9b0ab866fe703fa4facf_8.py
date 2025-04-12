import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostClassifier
import joblib


development_data = pd.read_csv(r"D:\Workspace\xianyu\Catboost_401_200\data.csv")

x = development_data[['Cl', 'GLU', 'DBil', 'LDH', 'BUN/SCr', 'CHE', 'IBil', 'UA', 'PDW', 'GGT', 'LY%']]
y = development_data['class']

other_params = {'iterations':400, 'depth': 3, 'l2_leaf_reg': 5,'random_state': 123}
model = CatBoostClassifier(**other_params,verbose=0)
model.fit(x, y)

model_path = 'd:\\Workspace\\xianyu\\Catboost_401_200\\model.pkl'
joblib.dump(model, model_path)
print(f"\n模型已保存至: {model_path}")

def predict_single_sample(Cl, GLU, DBil, LDH, BUN_SCr, CHE, IBil, UA, PDW, GGT, LY_pct):
    input_data = pd.DataFrame([[Cl, GLU, DBil, LDH, BUN_SCr, CHE, IBil, UA, PDW, GGT, LY_pct]], 
                              columns=['Cl', 'GLU', 'DBil', 'LDH', 'BUN/SCr', 'CHE', 'IBil', 'UA', 'PDW', 'GGT', 'LY%'])
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    return prediction[0], prediction_proba[0]


if __name__ == '__main__':

    test_result, test_proba = predict_single_sample(
        Cl=100,     
        GLU=5.5,
        DBil=10,
        LDH=200,
        BUN_SCr=20,
        CHE=8000,
        IBil=8,
        UA=300,
        PDW=12,
        GGT=30,
        LY_pct=25
    )
    
    print(f"\n预测结果: {test_result}")
    print(f"预测概率: {test_proba}")
