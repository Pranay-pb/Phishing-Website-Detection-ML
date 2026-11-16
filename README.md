\# Phishing Website Detection Using Machine Learning



\## Overview



Phishing Website Detection Using Machine Learning is a machine learning-based project designed to identify potentially malicious websites by analysing URL characteristics and applying trained classification algorithms.



The system extracts relevant features from URLs and uses machine learning models to distinguish between phishing and legitimate websites. It also incorporates model explainability to provide insights into the factors influencing classification decisions.



The project includes a Streamlit-based web application that allows users to submit URLs and view the model's predictions and associated probability estimates.



\## Key Features



\- URL-based feature extraction and preprocessing.

\- Detection of phishing and legitimate URLs using machine learning.

\- Implementation and comparison of multiple classification algorithms.

\- Random Forest and XGBoost model integration.

\- SHAP-based model explainability.

\- Interactive Streamlit dashboard.

\- Prediction probabilities for phishing and legitimate classes.



\## Machine Learning Algorithms



The project explores and compares six classification algorithms:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest

4\. Support Vector Machine (SVM)

5\. XGBoost

6\. Multilayer Perceptron (MLP)



Random Forest and XGBoost models are included in the project artifacts.



\## Technology Stack



\- \*\*Programming Language:\*\* Python

\- \*\*Data Processing:\*\* Pandas, NumPy

\- \*\*Machine Learning:\*\* Scikit-learn, XGBoost

\- \*\*Explainable AI:\*\* SHAP

\- \*\*Web Application:\*\* Streamlit

\- \*\*Model Serialization:\*\* Joblib

\- \*\*Development Environment:\*\* Jupyter Notebook, Visual Studio Code

\- \*\*Version Control:\*\* Git and GitHub



\## Project Structure



```text

Phishing-Website-Detection-ML/

│

├── dashboard/

│   └── app.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   ├── feature\_columns.pkl

│   ├── random\_forest\_model.pkl

│   └── xgboost\_model.pkl

│

├── notebooks/

│   ├── 01\_dataset\_exploration.ipynb

│   └── 02\_model\_training.ipynb

│

├── src/

│   └── feature\_extraction.py

│

├── .gitattributes

├── README.md

└── requirements.txt

```



\## Dataset



The project uses a dataset containing URL records labelled as phishing or legitimate websites.



The dataset undergoes preprocessing, feature extraction, duplicate analysis, and conflict handling before being used for model development and evaluation.



The target encoding is:



\- `0` – Phishing

\- `1` – Legitimate



\## Explainable AI



SHAP (SHapley Additive exPlanations) is used to interpret model predictions and understand the contribution of individual URL features.



This provides greater transparency into the classification process and helps analyse how different URL characteristics influence model decisions.



\## Running the Application Locally



Clone the repository:



```bash

git clone https://github.com/Pranay-pb/Phishing-Website-Detection-ML.git

```



Navigate to the project directory:



```bash

cd Phishing-Website-Detection-ML

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



Launch the Streamlit application:



```bash

streamlit run dashboard/app.py

```



\## Disclaimer



This project is developed for educational and research purposes. Its predictions are based on extracted URL characteristics and learned patterns in the training data.



A legitimate prediction does not guarantee that a website is safe, and a phishing prediction should not be treated as definitive proof of malicious activity.



\## Author



\*\*Pranay Bhardwaj\*\*



GitHub: \[Pranay-pb](https://github.com/Pranay-pb)



\## Project Objective



The objective of this project is to explore the application of machine learning and explainable artificial intelligence in identifying potentially malicious URLs, while developing an interactive system for accessible phishing detection.



