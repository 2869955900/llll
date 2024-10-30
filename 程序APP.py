import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 加载六分类的 XGBoost 模型
model = joblib.load('XGBoost.pkl')

# 定义特征名称
feature_names = [
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", 
    "Q10", "Q11", "Q12", "Q13", "Q14"
]

# 根据提供的最小值和最大值定义每个特征的输入范围
feature_ranges = {
    "Q1": (3.5, 24101.0),
    "Q2": (1.7, 5813.0),
    "Q3": (0.3, 1823.0),
    "Q4": (0.6, 3671.0),
    "Q5": (0.0, 419.1),
    "Q6": (10.5, 10486.0),
    "Q7": (0.0, 10.265306),
    "Q8": (0.047805, 16.410359),
    "Q9": (0.048232, 24.333333),
    "Q10": (0.009714, 17.098468),
    "Q11": (0.002443, 0.787330),
    "Q12": (0.010334, 0.745614),
    "Q13": (0.0, 0.759936),
    "Q14": (0.057795, 0.946897)
}

# Streamlit 用户界面
st.title("六分类预测模型")

# 为每个特征创建输入字段，并根据最小-最大值范围设置
feature_values = []
for feature, (min_val, max_val) in feature_ranges.items():
    value = st.number_input(f"{feature}:", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    feature_values.append(value)

# 将特征转换为 NumPy 数组以便模型预测
features = np.array([feature_values])

if st.button("预测"):
    # 进行六分类预测，并获取每个类别的概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class}")
    st.write("**每个类别的预测概率:**")
    for i, proba in enumerate(predicted_proba):
        st.write(f"类别 {i}: {proba * 100:.2f}%")

    # 计算 SHAP 值，并分别提取每个类别的 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 提取每个类别的 SHAP 值
    shap_values_class_0 = shap_values[:, :, 0]
    shap_values_class_1 = shap_values[:, :, 1]
    shap_values_class_2 = shap_values[:, :, 2]
    shap_values_class_3 = shap_values[:, :, 3]
    shap_values_class_4 = shap_values[:, :, 4]
    shap_values_class_5 = shap_values[:, :, 5]

    # 根据预测的类别选择对应的 SHAP 值
    shap_values_for_predicted_class = [
        shap_values_class_0, shap_values_class_1, shap_values_class_2,
        shap_values_class_3, shap_values_class_4, shap_values_class_5
    ][predicted_class]

    # 绘制预测类别的 SHAP 力图
    plt.figure()
    shap.force_plot(
        explainer.expected_value[predicted_class], 
        shap_values_for_predicted_class[0],  # 选择单个实例的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    # 在 Streamlit 中显示保存的图片
    st.image("shap_force_plot.png")
