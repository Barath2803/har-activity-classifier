# Human Activity Recognition with Five Classifiers

**Live app:** https://har-activity-classifier-cjizfycjxtpyyz4brv8hwu.streamlit.app/

## a. Problem statement
Classify what a person is doing (walking, climbing stairs, descending stairs, sitting, standing or lying) from the signals of a waist-mounted smartphone's accelerometer and gyroscope. This is a **6-class classification** problem. Five classical models are trained on the same data, compared on six metrics, and exposed through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Human Activity Recognition Using Smartphones* (Anguita et al., 2013).
- **Size:** 10,299 instances, **561 features** (time- and frequency-domain statistics such as mean, std, energy, entropy and correlations of body/gravity acceleration and gyroscope signals). All features are numeric and normalised to [-1, 1].
- **Target:** `Activity` with 6 classes: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.
- **Split:** the dataset's official subject-wise split: 7,352 training rows (`data/train_data.csv.gz`) and 2,947 test rows (`test_data.csv`). No subject appears in both sets.
- **Pre-processing:** the 561 raw columns are used as-is. Duplicate feature names in the original file were made unique with a numeric suffix. `StandardScaler` is applied inside the pipelines of Logistic Regression, kNN and SVM. Decision Tree and Naive Bayes use the raw values.

## c. GitHub repository
https://github.com/Barath2803/har-activity-classifier

```
har-activity-classifier/
├── app.py                 Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv          held-out test set (2,947 rows) to upload in the app
├── data/train_data.csv.gz training set
└── model/
    ├── prepare_data.py    builds the CSVs from the raw UCI folder
    ├── train_models.py    trains and saves the five models, writes metrics.csv
    ├── metrics.py         metric helper shared by training and app
    └── *.joblib           saved models
```

Run locally: `pip install -r requirements.txt && streamlit run app.py`. Retrain with `python model/train_models.py`.

## d. Models used

All metrics are on the 2,947-row test set. AUC is one-vs-rest macro-averaged; Precision, Recall and F1 are weighted averages across classes.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9545 | 0.9975 | 0.9564 | 0.9545 | 0.9545 | 0.9458 |
| Decision Tree | 0.8677 | 0.9169 | 0.8686 | 0.8677 | 0.8671 | 0.8413 |
| kNN | 0.8887 | 0.9818 | 0.8960 | 0.8887 | 0.8877 | 0.8679 |
| Naive Bayes | 0.7703 | 0.9578 | 0.7947 | 0.7703 | 0.7688 | 0.7286 |
| Support Vector Machine | 0.9542 | 0.9977 | 0.9549 | 0.9542 | 0.9540 | 0.9451 |

Settings: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=12`), kNN (`k=7`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=10`, `probability=True`); `random_state=42` throughout.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy (95.5%) and MCC. The 561 engineered features are close to linearly separable, so a linear model suffices. Its only notable error is SITTING predicted as STANDING (58 cases). |
| Decision Tree | 86.8% accuracy, the weakest AUC (0.917). A single tree splits on one feature at a time and overfits the many correlated features. It confuses sitting with standing and walking upstairs with walking. |
| kNN | 88.9% accuracy with a high AUC (0.982). Distances in 561 dimensions are less informative, so it mixes up sitting/standing and the three walking classes. It is also the slowest at prediction time, and its saved model is the largest because it stores the training set. |
| Naive Bayes | Lowest accuracy (77.0%) but a reasonable AUC (0.958). Its independence assumption fails badly on these highly correlated features, so its probabilities rank well while its hard decisions are often wrong. It misclassifies 211 LAYING samples as SITTING. |
| Support Vector Machine | 95.4% accuracy, within 0.03 points of Logistic Regression, and the highest AUC (0.9977). The RBF kernel separates the walking classes best (27 downstairs→upstairs errors versus 47 for the tree). |
| **Overall winner for this dataset** | **Logistic Regression** has the best Accuracy, Precision, Recall, F1 and MCC, and is the smallest and fastest to train and serve. SVM is a statistical tie, ahead only on AUC. |

## Streamlit app
- Upload a CSV of test data (use `test_data.csv`) in the sidebar.
- Pick a model from the dropdown.
- See Accuracy, AUC, Precision, Recall, F1 and MCC, plus the confusion matrix and classification report.
- The "Compare all models" tab shows the five models side by side on the uploaded file.
