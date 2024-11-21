# Breast Cancer Risk Prediction

This project aims to develop a predictive system to classify breast cancer tumors based on data derived from Fine Needle Aspiration (FNA) tests. Using machine learning techniques, the model predicts whether a tumor is malignant (cancerous) or benign (non-cancerous), providing crucial insights into early cancer detection.

![Brest Cancer Risk Prediction](https://github.com/kurmaviswakanth/Breast-Cancer-Risk-Prediction/blob/main/image.jpg)

## Project Overview

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). It contains features computed from digitized images of FNA tests, which describe the characteristics of the cell nuclei present in the breast tissue samples.

## Key Steps in the Project

- **Data Preprocessing:**
  - Removed irrelevant columns, such as the id column, and addressed missing values to ensure clean data.
  - Transformed categorical variables, like the diagnosis column, into numerical formats (malignant tumors coded as 1 and benign as 0) to make the data suitable for machine learning models.

- **Exploratory Data Analysis (EDA):**
  - Conducted a detailed analysis of key features such as tumor radius, texture, perimeter, area, and smoothness.
  - Identified features most correlated with the diagnosis, which were later used in model training.
  - Visualized data utilizing histograms, pair plots, and correlation matrices to understand the relationships between features.

- **Feature Engineering:**
  - Selected only the most relevant features for building the predictive model, removing redundant or less useful attributes.

- **AI-Powered Machine Learning Models:**
  - Employed several machine learning algorithms integrated with AI techniques for tumor classification, including:
    - Logistic Regression
    - Support Vector Machines (SVM)
    - Random Forest Classifier
    - Decision Trees
  - Applied hyperparameter tuning to optimize model performance, improving accuracy and generalization ability.


## AI-Enhanced Performance
After model tuning, the **Logistic Regression** and **Support Vector Machine (SVM)** models achieved an impressive accuracy rate of **97.08%**, exceeding the general industry benchmark of **95%** accuracy for breast cancer prediction models. This exceptional accuracy reflects the models' capability to reliably differentiate between benign and malignant tumors, potentially aiding medical professionals in making informed decisions.

## Tools and Technologies
The project was implemented using Python, utilizing the following libraries and tools:

- **Pandas** and **NumPy** for data manipulation and preprocessing.
- **Matplotlib** and **Seaborn** for data visualization and exploratory analysis.
- **Scikit-learn** for building and evaluating machine learning models.
- **TensorFlow** or **Keras** could also be integrated for deep learning approaches if needed.
- **Google Colab** for cloud-based code execution and collaboration.

## Conclusion

This project successfully demonstrates the application of artificial intelligence in healthcare, particularly in predicting breast cancer. By employing digitized FNA test data and AI-implemented machine learning algorithms, this system offers a promising tool for early diagnosis, which can significantly enhance treatment outcomes and potential survival rates for patients.
