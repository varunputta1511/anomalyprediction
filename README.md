# 🚀 NASA Turbofan Engine Predictive Maintenance & Anomaly Detection

## 📌 Project Overview
This project focuses on *predictive maintenance and anomaly detection* for turbofan engines using *machine learning (Random Forest, XGBoost, SVM)* and *deep learning (LSTM, RNN)* models.

We leverage the *NASA CMAPSS dataset, which contains **sensor data from 100+ engines operating under different conditions. Our goal is to predict the **Remaining Useful Life (RUL)* of engines and detect anomalies before failure occurs.

---

## 📂 Dataset: NASA C-MAPSS

The *C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset* consists of *run-to-failure* sensor readings from *turbofan engines* under varying operational conditions.

### 🔹 *Dataset Structure*
Each engine has *21 sensor readings* collected over time until failure. The dataset includes:
- *Unit Number (unit_nr)* → Unique engine ID.
- *Time (time_cycles)* → Operational cycle count.
- *Operational Settings (setting_1, setting_2, setting_3)* → Control parameters affecting engine conditions.
- *21 Sensor Measurements (s_1 to s_21)* → Temperature, pressure, speed, and efficiency indicators.
- *Remaining Useful Life (RUL)* → Number of cycles before failure.

### 🔹 *Preprocessing Steps*
✔ *Feature Selection* → Identified most relevant sensors using *correlation analysis*.
✔ *Normalization* → Scaled features for better model performance.
✔ *Time-Series Transformation* → Used *sliding window approach* for LSTM models.

---

## 📒 Implemented Notebooks

### 🧠 *1️⃣ Predictive Maintenance using LSTM*
📌 *Goal:* Build a *Long Short-Term Memory (LSTM) neural network* to predict *RUL*.
🔹 *Techniques Used:*
✔ Time-series data transformation for sequential learning.
✔ Multi-layer *LSTM model* with dropout to prevent overfitting.
✔ Optimized *batch size, learning rate, and epochs* for performance.

---

### 🌲 *2️⃣ Anomaly Detection using Random Forest*
📌 *Goal:* Detect anomalies in sensor data using *Random Forest* classification.
🔹 *Techniques Used:*
✔ *Feature importance analysis* to identify key contributing sensors.
✔ *Random Forest model* to classify normal vs. failing engines.
✔ *Hyperparameter tuning* (n_estimators, max_depth) for improved accuracy.

---

### 🤖 *3️⃣ Implementation of 10 Different Models for Best RUL Prediction*
📌 *Goal:* Compare *10 different regression models* to find the best one for RUL prediction.
🔹 *Implemented Models:*
1️⃣ *Linear Regression*
2️⃣ *Lasso Regression*
3️⃣ *Ridge Regression*
4️⃣ *Decision Tree*
5️⃣ *Random Forest*
6️⃣ *Support Vector Regressor (SVR)*
7️⃣ *Gradient Boosting Regressor*
8️⃣ *Artificial Neural Network (ANN)*
9️⃣ *Recurrent Neural Network (RNN)*
🔟 *Long Short-Term Memory (LSTM)*

🔹 *Findings:*
✔ *SVR & LSTM outperformed traditional models.*
✔ *Random Forest achieved strong baseline results.*
✔ *Decision Trees overfitted, showing high training accuracy but poor test performance.*

---

## 📊 *Model Comparison & Results*

| Model                          | MSE   | RMSE  | R² Score | Accuracy (%) |
|--------------------------------|-------|-------|----------|-------------|
| *Linear Regression*          | 6992  | 83.6  | 3.08     | 56          |
| *Lasso Regression*           | 6841  | 82.7  | 2.99     | 56          |
| *Ridge Regression*           | 6992  | 83.6  | 3.08     | 56          |
| *Decision Tree*              | 10101 | 100.5 | 4.89     | 100 (Overfitting) |
| *Random Forest*              | 7279  | 85.3  | 3.25     | 94          |
| *Support Vector Regressor*   | 5770  | 75.9  | 2.36     | 57 (Best Traditional) |
| *Gradient Boosting*          | 6968  | 83.4  | 3.06     | 60          |
| *Artificial Neural Network*  | 7341  | 85.6  | 3.29     | 72.89       |
| *Recurrent Neural Network*   | 6648.1| 81.53 | 2.88     | 100         |
| *LSTM (Deep Learning)*       | 6711.7| 81.92 | 2.91     | 100 (Best Deep Learning) |

📌 *Key Findings:*
✔ *Support Vector Regressor (SVR) had the lowest RMSE (75.9), making it the best traditional model.*
✔ *LSTM showed promising results with time-series learning capabilities.*
✔ *Decision Tree overfitted, performing poorly on unseen data.*
✔ *Random Forest provided a strong baseline with 94% accuracy.*

---

## 🎨 Streamlit Application for RUL Prediction

We developed a *Streamlit web application* that allows users to *input sensor data* and predict the *Remaining Useful Life (RUL)* of an engine in real-time. The application:
✔ Loads the *trained Random Forest model* for predictions.
✔ Accepts *manual input or CSV uploads*.
✔ Provides *interactive visualizations* for better interpretability.

To run the application:
bash
streamlit run app.py


---

## 🛠️ *Installation & Usage*

### 📥 *Setup the Environment*
bash
git clone https://github.com/your-username/turbofan-predictive-maintenance.git  
cd turbofan-predictive-maintenance  
pip install -r requirements.txt  


### 🏃 *Run Notebooks*
Execute Jupyter Notebooks in the notebooks/ directory:
bash
jupyter notebook


---

## 🚀 *Future Improvements*
✔ *Experiment with XGBoost & LightGBM* for better predictive performance.  
✔ *Implement Transformer models for time-series forecasting.*  
✔ *Deploy model using Flask or FastAPI for real-time predictions.*  
✔ *Feature engineering to improve sensor data representation.*  

---

## 📜 *References*
- 📄 [NASA C-MAPSS Dataset](https://data.nasa.gov/Aerospace/NASA-C-MAPSS-Dataset/)  
- 📄 [Predictive Maintenance & Machine Learning](https://arxiv.org/pdf/1709.05603.pdf)  
- 📄 [LSTMs for Time Series Prediction](https://www.tensorflow.org/tutorials/structured_data/time_series)  

---

## 🤝 *Contributors*
👨‍💻 *Siddhant Shetty* ([@siddhantshetty](https://github.com/Siddhantshetty)) 
👨‍💻 *Varun Putta* ([@varunputta1511](https://github.com/varunputta1511))  
📩 Feel free to *open issues & pull requests* for improvements!
