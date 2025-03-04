import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle 

# Load model and scaler
sc = StandardScaler()
file = open(r'final_model.pkl', 'rb')
model = pickle.load(file)
rf = RandomForestRegressor()

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔧 Process Overview")
st.sidebar.markdown("1️⃣ Upload your training, test, and target datasets.")
st.sidebar.markdown("2️⃣ Preprocess the data (drop unwanted columns, normalize features, and compute RUL).")
st.sidebar.markdown("3️⃣ Train the Random Forest model and generate predictions.")
st.sidebar.markdown("4️⃣ Visualize actual vs predicted results.")
st.sidebar.markdown("5️⃣ Download predictions for further analysis.")

@st.cache_resource
def predict(x_train, y_train, x_test, y_test):
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test).reshape(-1, 1)
    
    st.subheader("📊 Actual vs Predicted RUL")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test, color="blue", linewidth=2, linestyle="-", label="Actual")
    ax.plot(y_pred, color='red', linewidth=2, linestyle="--", label="Predicted")
    ax.legend()
    st.pyplot(fig)
    
    mse = mean_squared_error(y_test, y_pred)
    acc = round(rf.score(x_train, y_train), 2) * 100  
    
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted RUL'])
    st.subheader("📜 Predictions Table")
    st.dataframe(y_pred_df, width=800, height=300)
    
    return mse, acc, y_pred_df.to_csv(index=False)


def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    df = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df.drop("max_cycle", axis=1, inplace=True)
    
    return df

def main():
    st.title("🚀 Predictive Maintenance Dashboard")
    st.image('https://upload.wikimedia.org/wikipedia/commons/4/49/Turbofan_operation_%28lbp%29.png', use_container_width=True)
    
    st.markdown("### Upload your datasets to begin the prediction process.")
    
    train_file = st.file_uploader("📂 Upload Training Data", type=["csv", "txt"])
    test_file = st.file_uploader("📂 Upload Test Data", type=["csv", "txt"])
    y_test_file = st.file_uploader("📂 Upload Target (y_test) Data", type=["csv", "txt"])
    
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    time_cycle_each_test = []
    
    if train_file is not None:
        train_df = pd.read_csv(train_file, sep='\s+' if train_file.type == "text/plain" else ',', header=None)
        train_df.columns = col_names
        st.success("✅ Training Data Uploaded Successfully!")
        
        train_df = add_remaining_useful_life(train_df)
        train_df.drop(['setting_3', 's_1', 's_10', 's_18', 's_19'], axis=1, inplace=True)
        train_df.drop_duplicates(inplace=True)
        
        X_train = train_df[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15', 's_17', 's_20', 's_21']]
        y_train = train_df[['RUL']]
        X_train = sc.fit_transform(X_train)
    
    if test_file is not None:
        test_df = pd.read_csv(test_file, sep='\s+' if test_file.type == "text/plain" else ',', header=None)
        test_df.columns = col_names
        st.success("✅ Test Data Uploaded Successfully!")
        test_df.drop(['setting_3', 's_1', 's_10', 's_18', 's_19'], axis=1, inplace=True)
        X_test = test_df[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15', 's_17', 's_20', 's_21']]
        X_test = sc.transform(X_test)
        
        for i in range(1, len(test_df.unit_nr.unique()) + 1):
            time_cycle_each_test.append(len(test_df.time_cycles[test_df['unit_nr'] == i]))
    
    if y_test_file is not None:
        y_test = pd.read_csv(y_test_file, sep='\s+' if y_test_file.type == "text/plain" else ',', header=None, names=['RUL'])
        st.success("✅ y_test Data Uploaded Successfully!")
        
        y_test_expanded = np.repeat(y_test['RUL'].values, repeats=time_cycle_each_test[:len(y_test['RUL'])])
        if len(y_test_expanded) != len(X_test):
            st.error(f"Size mismatch! y_test: {len(y_test_expanded)}, X_test: {len(X_test)}")
            return
        y_test = pd.DataFrame({'RUL': y_test_expanded})
    
        if st.button('🚀 Predict RUL'):
            mse, acc, y_pred_csv = predict(X_train, y_train, X_test, y_test)
            st.download_button(label='📥 Download Predictions', data=y_pred_csv, file_name='predictions.csv')   
            st.success(f'🎯 Model Performance: MSE = {mse:.2f}, Accuracy = {acc:.2f}%')

if __name__ == "__main__":
    main()
