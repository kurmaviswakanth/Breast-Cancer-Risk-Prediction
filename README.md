# Breast Cancer Risk Prediction

This project aims to develop a predictive system to classify breast cancer tumors based on data derived from Fine Needle Aspiration (FNA) tests. Using machine learning techniques, the model predicts whether a tumor is malignant (cancerous) or benign (non-cancerous), providing crucial insights into early cancer detection.

![Brest Cancer Risk Prediction](https://github.com/kurmaviswakanth/Breast-Cancer-Risk-Prediction/blob/main/image.jpg)

## Project Overview

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). It contains features computed from digitized images of FNA tests, which describe the characteristics of the cell nuclei present in the breast tissue samples.

### Key Steps in the Project

- **Data Preprocessing:**
  - Removed irrelevant columns such as the id column, and handled missing values to ensure clean data.
  - Categorical variables like the diagnosis column were converted into numerical formats, with malignant tumors labeled as 1 and benign as 0, making it suitable for machine learning models.

- **Exploratory Data Analysis (EDA):**
  - Conducted a thorough analysis of key features such as tumor radius, texture, perimeter, area, and smoothness.
  - This step identified the features most correlated with the diagnosis, which were later used in model training.
  - The data was visualized using histograms, pair plots, and correlation matrices to understand the relationships between features.

- **Feature Engineering:**
  - After analyzing the dataset, only the most relevant features were kept for building the predictive model, removing redundant or less useful attributes.

- **Machine Learning Models:**
  - Several machine learning algorithms were applied to classify the tumors, including:
    - Logistic Regression
    - Support Vector Machines (SVM)
    - Random Forest Classifier
    - Decision Trees
  - After training these models, hyperparameter tuning was performed to optimize performance, improving the model's accuracy and generalization ability.

### Performance

After model tuning, the **Logistic Regression** and **Support Vector Machine (SVM)** models achieved the best results, with an accuracy of **97.08%**, surpassing the general industry benchmark of **95%** accuracy for breast cancer prediction models. This high accuracy reflects the models' ability to reliably distinguish between benign and malignant tumors.

## Tools and Technologies

The project was implemented using Python, with the following libraries and tools:

- **Pandas** and **NumPy** for data manipulation and preprocessing.
- **Matplotlib** and **Seaborn** for data visualization and exploratory analysis.
- **Scikit-learn** for building and evaluating machine learning models.
- **Google Colab** for cloud-based code execution and collaboration.

## Conclusion

This project successfully demonstrates the application of machine learning in healthcare, specifically in predicting breast cancer. By using digitized FNA test data and machine learning algorithms, this system offers a potential tool for early diagnosis, which can significantly improve treatment outcomes.
