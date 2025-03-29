### Credit Card Fraud Detection: A Machine Learning Approach
### DATASET LINK: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
### Objective
The primary objective of this project is to detect fraudulent credit card transactions using statistical analysis and machine learning models. By leveraging data preprocessing techniques, exploratory data analysis (EDA), and various classification algorithms, we aim to automate fraud detection with high accuracy and reliability.

### Key Implementations
Throughout this project, the following tasks have been performed:

**Data Cleaning**: Handling missing values and removing duplicates.

**Exploratory Data Analysis (EDA)**: Understanding data distribution, feature relationships, and target class imbalance.

**Statistical Analysis**: Evaluating data characteristics to determine the best-suited algorithm.

**Machine Learning Model Development**: Training and optimizing models for fraud detection.

**Evaluation Metrics**: Assessing model performance using multiple key performance indicators (KPIs).

### Performance Indicators
To ensure model effectiveness, the following metrics are considered:

Accuracy, Precision, Recall, F1 Score (Measure predictive performance)

Mean Squared Error (MSE), Mean Absolute Error (MAE) (Assess prediction errors)

AUC-ROC (Area Under the Receiver Operating Characteristic Curve)

AUC-PR (Area Under the Precision-Recall Curve)

Cross-Validation Score (Ensures model generalization)

### Data Preprocessing Steps
**1. Understanding the Dataset**
A Pandas DataFrame is created to inspect the dataset's structure, including the number of rows, columns, and data types.

The target variable is identified, representing fraudulent and non-fraudulent transactions.

**2.Data Cleaning**
Handling Missing Values: Various strategies are available for dealing with missing data. Fortunately, this dataset does not contain null values.

Removing Duplicates: Any duplicate records are identified and dropped immediately to maintain data integrity.

### Feature Transformation:

All features must be in numerical format (integer or float) for machine learning models.

If categorical variables were present, One-Hot Encoding would be used to convert them into numerical representations. However, this dataset does not contain categorical features.

**3. Exploratory Data Analysis (EDA)**
Using the .describe() method in Pandas, the following statistical insights are obtained:

Count: Number of non-null values.

Mean: Average value of each feature.

Standard Deviation (std): Measures the spread of values from the mean.

Min and Max: Minimum and maximum values in the dataset.

Quartiles (Q1, Q2, Q3): Represent the 25th, 50th (median), and 75th percentiles.

**4.Handling Class Imbalance**
The dataset is examined for class imbalance by visualizing the distribution of fraudulent (Class 1) and non-fraudulent (Class 0) transactions using Seaborn.

Since fraud cases are significantly less frequent than legitimate transactions, techniques such as SMOTE (Synthetic Minority Over-sampling Technique) and Random Undersampling are considered to balance the dataset.

**5. Visualizing Key Features**
The distribution of Transaction Time and Transaction Amount is analyzed to identify potential trends.

A correlation heatmap using Seaborn is plotted to understand relationships between variables, helping in feature selection for model training.

By following these structured steps, the dataset is effectively prepared for training machine learning models that can accurately detect fraudulent transactions.

### Handling Class Imbalance
Class imbalance is a critical issue in fraud detection, as fraudulent transactions are significantly fewer than legitimate ones. To ensure the model learns effectively, several techniques are employed to balance the dataset:

Oversampling (SMOTE - Synthetic Minority Over-sampling Technique)

Generates synthetic fraud cases to increase the representation of the minority class.

Undersampling

Reduces the number of majority class (non-fraud transactions) to match the minority class.

Class Weight Adjustment

Assigns higher weights to fraudulent transactions during model training to compensate for the imbalance.

Anomaly Detection Models

Techniques such as Isolation Forest and Autoencoders learn normal transaction patterns and detect fraud as deviations from these patterns.

### Implementation of Class Balancing Strategy
To resolve class imbalance, a combination of SMOTE (Synthetic Minority Over-sampling Technique) and Random Undersampling is applied:

First, SMOTE is used to generate synthetic fraud samples.

Next, Random Undersampling reduces the number of majority-class samples.

The process is repeated, applying undersampling followed by SMOTE to achieve a more refined balance.

The final resampled dataset is converted into a Pandas DataFrame and visualized using a count plot to confirm the balanced class distribution.

With the dataset now balanced, we move on to the next step—model building.

### Model Selection and Planning
Before selecting the best model for fraud detection, it is crucial to analyze different machine learning algorithms and their suitability for this dataset. Below is a list of models considered, along with their characteristics and applicability:

### ✅ Logistic Regression
Conditions to Apply:

Features should not be highly correlated (verify using a correlation matrix).

Works well with a balanced dataset (hence, resampling techniques like SMOTE are applied).

Interpretable but may struggle with complex relationships.

### ✅ Random Forest
Conditions to Apply:

Handles imbalanced data well, but class weighting may be required.

No need for feature scaling.

Computationally expensive for large datasets.

### ✅ XGBoost (Extreme Gradient Boosting)
Conditions to Apply:

Performs well on imbalanced datasets (tuning scale_pos_weight helps).

Requires hyperparameter tuning for optimal performance.

Can handle missing values automatically.

### ✅ Support Vector Machine (SVM)
Conditions to Apply:

Works well for low-dimensional datasets (feature extraction via PCA may be required).

Kernel selection is crucial (rbf kernel is useful for non-linear patterns).

Computationally expensive for large datasets.

### ✅ K-Nearest Neighbors (KNN)
Conditions to Apply:

Suitable for small to medium-sized datasets.

Requires feature scaling (standardization or normalization).

Sensitive to class imbalance, so resampling techniques are applied.

### ✅ Naïve Bayes
Conditions to Apply:

Assumes feature independence (not ideal for highly correlated features).

Computationally efficient but may struggle with non-linear relationships.

### Model Training
The following machine learning models were trained for fraud detection:

Logistic Regression

Random Forest

XGBoost

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naïve Bayes

With these models implemented, the next step is to evaluate their performance using various key performance indicators (KPIs) to determine the best approach for fraud detection.

### Model Performance & Accuracy

After training and evaluating the models, the following accuracy scores were obtained:

Logistic Regression: 0.9793

Random Forest: 0.9999

XGBoost: 0.9998

Support Vector Machine (SVM): 0.9966

K-Nearest Neighbors (KNN): 0.9992

Naïve Bayes: 0.9239

### Conclusion

The results indicate that Random Forest and XGBoost performed exceptionally well, achieving near-perfect accuracy. While Logistic Regression and SVM also performed well, Naïve Bayes had the lowest accuracy, likely due to its assumption of feature independence.

For real-world fraud detection, an ensemble approach combining Random Forest and XGBoost, along with robust anomaly detection techniques, can enhance model reliability and generalization.
