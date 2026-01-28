import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Algerian Forest Fires", layout="wide")
st.title("🔥 Algerian Forest Fires Analysis & Prediction")

# ===============================
# Dataset Info
# ===============================
with st.expander("📄 Dataset Information"):
    st.markdown("""
    - 244 instances (Bejaia & Sidi Bel-Abbes)
    - Period: June–September 2012
    - Target: Fire / Not Fire
    - Weather + FWI indices
    """)

# ===============================
# Load Dataset
# ===============================
st.subheader("📂 Load Dataset")

@st.cache_data
def load_data():
    return pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv", header=1)

df = load_data()
st.dataframe(df.head())

# ===============================
# Data Cleaning
# ===============================
st.subheader("🧹 Data Cleaning")

df.loc[:122, "Region"] = 0
df.loc[122:, "Region"] = 1
df["Region"] = df["Region"].astype(int)

df = df.dropna().reset_index(drop=True)
df = df.drop(122).reset_index(drop=True)

df.columns = df.columns.str.strip()
df[['month','day','year','Temperature','RH','Ws']] = df[['month','day','year','Temperature','RH','Ws']].astype(int)

objects = [col for col in df.columns if df[col].dtype == 'O']
for col in objects:
    if col != 'Classes':
        df[col] = df[col].astype(float)

st.success("✅ Data cleaned successfully")
st.dataframe(df.head())

# ===============================
# Feature Engineering
# ===============================
df_copy = df.drop(['day','month','year'], axis=1)

df_copy['Classes'] = np.where(
    df_copy['Classes'].str.contains('not fire'), 0, 1
)

st.subheader("📊 Class Distribution")
st.write(df_copy['Classes'].value_counts())

# ===============================
# Visualization
# ===============================
st.subheader("📈 Visualizations")

fig1 = plt.figure(figsize=(10,5))
df_copy.hist(bins=30)
st.pyplot(fig1)

fig2 = plt.figure(figsize=(6,6))
percentage = df_copy['Classes'].value_counts(normalize=True) * 100
plt.pie(percentage, labels=["Fire", "Not Fire"], autopct='%1.1f%%')
plt.title("Fire Distribution")
st.pyplot(fig2)

fig3 = plt.figure(figsize=(12,6))
sns.heatmap(df_copy.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig3)

# ===============================
# Model Training
# ===============================
st.subheader("🤖 Model Training")

X = df_copy.drop('FWI', axis=1)
y = df_copy['FWI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "RidgeCV": RidgeCV(cv=5),
    "ElasticNet": ElasticNet()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.markdown(f"### 🔹 {name}")
    st.write("R2 Score:", r2_score(y_test, y_pred))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))

    fig = plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    st.pyplot(fig)

st.success("🎉 App successfully converted from Colab to Streamlit!")
