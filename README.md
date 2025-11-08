# Heart Disease Prediction using Machine Learning

This project uses clinical data and machine learning to predict the likelihood of a person having heart disease. The aim is to assist in early diagnosis using simple input features and a Logistic Regression model.

---

##  Features Included

- Logistic Regression ML model
- Performance metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Outlier detection using boxplots
- ROC Curve plotted for performance visualization
- Trained model exported as `.pkl` using `joblib`
- Predicted results converted into readable class labels

---

## Sample Metrics

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 0.69%    |
| Precision    | 0.52%    |
| Recall       | 1.0%    |
| F1 Score     | 0.68%    |
| ROC-AUC      | 0.94%    |

---

## Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib`

---

##  Dataset Info

- Source: Kaggle – https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog
- Contains health indicators like age, cholesterol, resting blood pressure, max heart rate, and more.
- Target column: `target` (0 = No Heart Disease, 1 = Heart Disease Present)

---
