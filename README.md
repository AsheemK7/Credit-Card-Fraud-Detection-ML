# Credit Card Fraud Detection Using Machine Learning

A Machine Learning project that identifies potentially fraudulent credit card transactions using classification algorithms. The project compares multiple ML models and evaluates their performance using precision, recall, F1-score, and accuracy.

## Project Overview

Credit card fraud is a major challenge in digital financial transactions. This project aims to detect fraudulent transactions using supervised Machine Learning techniques while addressing the challenge of highly imbalanced datasets.

Three classification algorithms were trained and evaluated to identify the best-performing model.

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Joblib
- Google Colab

## Machine Learning Models

The following algorithms were implemented:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier

## Model Performance

The models were evaluated on the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.37% | 5.30% | 87.37% | 10.00% |
| Random Forest | **99.95%** | **91.25%** | 76.84% | **83.43%** |
| Decision Tree | 99.74% | 35.06% | 64.21% | 45.35% |

### Best Performing Model: Random Forest

Random Forest achieved the best overall balance between precision and recall among the three evaluated models.

- Accuracy: 99.95%
- Precision: 91.25%
- Recall: 76.84%
- F1-Score: 83.43%

**Note:** Due to class imbalance, accuracy alone is not a reliable indicator of fraud detection performance. Precision, recall and F1-score are particularly important.

## Feature Importance

Random Forest feature importance identified the following top contributing features:

| Feature | Importance |
|---|---:|
| V14 | 0.191439 |
| V10 | 0.118815 |
| V12 | 0.101301 |
| V17 | 0.093116 |
| V4 | 0.088793 |

V14 was the most influential feature in the trained Random Forest model.

## Project Structure

```text
Credit-Card-Fraud-Detection-ML/
│
├── Credit_Card_Fraud_Detection.ipynb
├── credit_fraud_model.pkl
├── credit_fraud_scaler.pkl
├── requirements.txt
└── README.md
```

## Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/AsheemK7/Credit-Card-Fraud-Detection-ML.git
```

2. Navigate to the project directory:

```bash
cd Credit-Card-Fraud-Detection-ML
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook or upload it to Google Colab.

5. Run the notebook cells to explore preprocessing, model training, evaluation and predictions.

## Saved Model

The trained Random Forest model and preprocessing scaler are saved using Joblib:

- `credit_fraud_model.pkl`
- `credit_fraud_scaler.pkl`

These files can be loaded for future predictions without retraining the model.

## Key Learnings

- Data preprocessing and feature scaling
- Handling imbalanced classification problems
- Training and comparing multiple ML algorithms
- Model evaluation using classification metrics
- Confusion matrix analysis
- Feature importance analysis
- Saving and reusing trained ML models

## Future Improvements

- Hyperparameter tuning
- Fraud probability threshold optimization
- Cross-validation
- Advanced ensemble techniques
- Deployment using Flask or FastAPI

## Author

**Asheem Khan**

Computer Science Undergraduate | Machine Learning | Data Science

- GitHub: https://github.com/AsheemK7
- LinkedIn: https://www.linkedin.com/in/asheem-khan-97864228b/

---

If you find this project useful, feel free to explore the repository.
