# Diabetes-prediction-using-Machine-Learning
🩺 Diabetes Prediction using Machine Learning
This project focuses on building a machine learning model to predict whether an individual has diabetes based on diagnostic measurements. The dataset is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, targeting Pima Indian women aged 21 and above.

📌 Objective
To develop an accurate model that classifies patients as diabetic or non-diabetic using various health metrics.

📂 Dataset Overview
Source: diabetes.csv

Observations: 768

Features: 8 predictors + 1 target (Outcome)

Features:
Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure

SkinThickness: Triceps skin fold thickness

Insulin: 2-Hour serum insulin

BMI: Body mass index

DiabetesPedigreeFunction: Genetic impact of diabetes

Age: Age in years

Outcome: 0 = Non-diabetic, 1 = Diabetic

⚙️ Machine Learning Workflow
Data Preprocessing and Cleaning

Exploratory Data Analysis (EDA)

Feature Selection

Model Training with:

Logistic Regression

Random Forest

Decision Tree

K-Nearest Neighbors (KNN)

Model Evaluation using Accuracy, Confusion Matrix, and ROC-AUC

Final Model Selection

🧪 Tools & Technologies
Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Visualization: Matplotlib, Seaborn

Jupyter Notebook

✅ Sample Output
Best Accuracy Achieved: Approximately 79–85% depending on the model

Confusion Matrix Output:
[[90 17]
 [14 49]]
 
Classification Report:
Precision: 0.75  
Recall: 0.78  
F1-score: 0.76  

📈 Key Insights
Glucose levels and BMI showed the strongest correlation with diabetes risk.

Models like Random Forest and Logistic Regression provided the most balanced performance.

📎 How to Run
Clone the repo
git clone https://github.com/your-username/diabetes-prediction.git

cd diabetes-prediction
Install dependencies
pip install -r requirements.txt

Run the Jupyter Notebook
jupyter notebook main-checkpoint.ipynb

