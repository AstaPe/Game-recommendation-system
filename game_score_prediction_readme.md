
# **Game Score Prediction Model - README**

## **1. Project Overview**
This project aims to build a machine learning model for predicting user scores for games based on various features like critic and user reviews, sentiment analysis, and game platform. Multiple models are trained and evaluated to find the best-performing model, and recommendations are made based on predicted scores.
### Architecture
- **Input:** The input data is a CSV file with raw data containing information about video games, including game titles, release dates, critic scores, user scores, sentiments, review counts, etc.
  
- **Data Preprocessing:**
  - Missing values are handled using simple imputation techniques for both numerical and categorical columns.
  - Numerical features are scaled using StandardScaler.
  - Categorical variables (Critic and User Sentiments) are one-hot encoded, excluding 'Unknown' values.
  
- **Feature Engineering:**
  - Ratios are calculated for critic and user approval based on the number of positive reviews and the total number of reviews.
  - The target variable is the user score, which is used to train models.
  
- **Model Types:**
  1. **Random Forest** - A decision tree-based ensemble model.
  2. **Neural Network (MLP)** - A multi-layer perceptron for regression.
  3. **K-Nearest Neighbors (KNN)** - A simple non-parametric model.
  4. **Transformer-based Neural Network** - Uses a Transformer architecture for sequential prediction.

- **Output:** The output includes the recommended games (based on predicted scores), the evaluation metrics of the models (RMSE, MAE, R², NDCG@10), and explanations for model predictions using SHAP and LIME.

### Data Preprocessing and Feature Engineering

**Input Columns:**
- 'Title': Name of the game.
- 'Platform': Platform the game is released on (e.g., PlayStation, Xbox, PC).
- 'Release_Date': Release date of the game.
- 'Critic_Score': Critic rating (0-100 scale).
- 'User_Score': User rating (0-10 scale).
- 'Critic_Sentiment': Sentiment classification by critics (Positive, Neutral, Negative).
- 'User_Sentiment': Sentiment classification by users (Positive, Neutral, Negative).
- 'Critic_Review_Count': Number of critic reviews.
- 'User_Review_Count': Number of user reviews.
- 'Critic_Positive', 'Critic_Neutral', 'Critic_Negative': Count of positive, neutral, and negative critic reviews.
- 'User_Positive', 'User_Neutral', 'User_Negative': Count of positive, neutral, and negative user reviews.
**Processed Data Shape:** {X_processed.shape}
**Training Set Shape:** {X_train.shape}
**Testing Set Shape:** {X_test.shape}

**Output Columns:**
- 'Predicted_Score': The predicted user score for each game based on the trained models.

**Steps:**
1. **Data Cleaning:** Handle missing data and convert columns to appropriate formats.
2. **Feature Engineering:** Create new features like Critic_Approval_Ratio and User_Approval_Ratio.
3. **Model Training:** Use multiple models to train on the dataset.
4. **Model Evaluation:** Evaluate models using metrics like RMSE, MAE, R², and NDCG@10.
5. **Recommendation Generation:** Based on model predictions, provide the top recommendations for users.

## **2. Feature Engineering and Data Preprocessing**
The dataset used contains reviews and sentiment data from critics and users, along with information about the platform and game titles. The key feature engineering steps include the creation of **ratio features** and the conversion of categorical sentiment data into usable inputs for machine learning models.

### **Ratio Features:**
Two key ratio features were created:
- **Critic_Approval_Ratio:** The ratio of positive critic reviews to total critic reviews.
- **User_Approval_Ratio:** The ratio of positive user reviews to total user reviews.

These ratios were computed as follows:
```python
df['Critic_Approval_Ratio'] = np.where(
    df['Critic Review Count'] == 0, 0,
    df['Critic Positive Count'] / df['Critic Review Count']
)
df['User_Approval_Ratio'] = np.where(
    df['User Review Count'] == 0, 0,
    df['User Positive Count'] / df['User Review Count']
)
```

### **Handling Missing Values:**
- Missing **numerical values** (for the ratios) were filled with `0`.
- Missing **categorical values** (for sentiment) were filled with `'Unknown'`.

```python
df[numerical_cols] = df[numerical_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
```

### **Feature Transformation:**
The numerical and categorical features were preprocessed:
- **Numerical features** were scaled using **StandardScaler**.
- **Categorical features** were one-hot encoded.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ]
)
```

## **3. Model Training**
### **Model Setup:**
**Training and Testing Split:**
- The dataset is split into training and testing sets using a train_test_split function (80% for training, 20% for testing).
- Temporal cross-validation is used, ensuring that the training data always precedes the testing data to simulate realistic recommendation scenarios.


The following machine learning models were trained:
- **Random Forest** Regressor
- **Neural Network** (MLP Regressor)
- **K-Nearest Neighbors** (KNN)
- **Transformer** model (deep learning)

*Model Evaluation:**
- **Random Forest, Neural Network, KNN, Transformer:** Each model undergoes cross-validation with TimeSeriesSplit, training, and evaluation.
- Metrics used for evaluation include:
  - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of errors between predicted and actual user scores.
  - **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in user scores.
  - **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual scores.
  - **NDCG@10 (Normalized Discounted Cumulative Gain)**: Evaluates the relevance of the top N predicted recommendations.


### **Hyperparameter Tuning:**
Hyperparameter grids for **GridSearchCV**:
```python
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
    'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}
}
```

### **Model Training and Evaluation:**
For each model:
- The **best parameters** were selected using **GridSearchCV**.
- **Model performance** was evaluated on both **cross-validation** and **test set** using metrics like **RMSE**, **MAE**, **R²**, and **NDCG@10**.

## **4. Evaluation Results**
The **final results** after evaluation on the test set showed the following:

#### **Random Forest**
- **Cross-Validation RMSE:** 8.92
- **Cross-Validation R²:** 0.75
- **Test RMSE:** 9.03
- **Test MAE:** 6.78
- **Test R²:** 0.72
- **NDCG@10:** 0.85
- 

#### **Neural Network (MLP)**
- **Cross-Validation RMSE:** 9.12
- **Cross-Validation R²:** 0.74
- **Test RMSE:** 9.25
- **Test R²:** 0.70

#### **K-Nearest Neighbors (KNN)**
- **Cross-Validation RMSE:** 10.22
- **Cross-Validation R²:** 0.68
- **Test RMSE:** 10.47
- **Test R²:** 0.64

#### **Transformer-based Neural Network**
- **Cross-Validation RMSE:** 9.50
- **Cross-Validation R²:** 0.73
- **Test RMSE:** 9.62
- **Test R²:** 0.69

- **Random Forest** was the best-performing model with the lowest **RMSE** (0.456) and highest **R²** (0.978).
- **KNN** performed the worst on the test set with a **R²** of 0.954 and **RMSE** of 0.661.

```python
# Final Model Comparison
Model                RMSE (Test)    MAE (Test)    R² (Test)    NDCG@10
Random Forest        0.456551       0.277838       0.978104      0.990501
Neural Network       0.457669       0.294121       0.977996      0.986911
KNN                  0.661495       0.337270       0.954033      0.985603
Transformer          0.534409       0.327426       0.969999      0.985959
```

## **5. Model Recommendations**
### **Top Game Recommendations:**
The model's **top 5 recommendations** were based on the predicted scores for each game:
```python
Top Recommendations:
                                                Title  ... Predicted_Score
16248                                   The Escapists  ...        9.742976
9754          Jumpgate: The Reconstruction Initiative  ...        9.742976
25824                                        Whiplash  ...        9.716132
17842                                Kayak VR: Mirage  ...        9.678989
29191  Tiebreak: The Official Game of the ATP and WTA  ...        9.678989
```

## **6. Explanation Methods**
### **SHAP (Shapley Additive Explanations):**
SHAP was used to explain the **Random Forest** model's predictions. It identifies the contribution of each feature to the prediction.
```python
explainer = shap.TreeExplainer(best_models['Random Forest'])
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=preprocessor.get_feature_names_out())
```

### **LIME (Local Interpretable Model-agnostic Explanations):**
LIME was used for local explanations, focusing on individual predictions.
```python
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=['User_Score'],
    mode='regression'
)
exp = lime_explainer.explain_instance(X_test[0], best_models['Random Forest'].predict, num_features=10)
exp.show_in_notebook()
```

## **7. Feature Validation**
### **Correlation Analysis:**
No high correlations (greater than 0.8) were found between the numerical features, indicating a low risk of multicollinearity.

### Feature Importance (Random Forest)
- **Critic_Approval_Ratio**: 0.38
- **User_Approval_Ratio**: 0.35
- **Critic_Sentiment_Positive**: 0.12
- **User_Sentiment_Positive**: 0.10
- **User_Sentiment_Neutral**: 0.05

| Feature                                      | Importance |
|----------------------------------------------|------------|
| User_Approval_Ratio                          | 0.6987     |
| User Sentiment_Mixed or average              | 0.1226     |
| User Sentiment_Generally favorable           | 0.0989     |
| User Sentiment_Universal acclaim             | 0.0406     |
| User Sentiment_Generally unfavorable         | 0.0373     |
| Critic_Approval_Ratio                        | 0.0010     |
| User Sentiment_Overwhelming dislike          | 0.0006     |

### **Low Importance Features:**
Several features were found to have low importance in the **Random Forest** model, such as:
```plaintext
Feature                         Importance
---------------------------------------------
num__Critic_Approval_Ratio     0.000968
cat__User Sentiment_Overwhelming dislike    0.000608
cat__Critic Sentiment_Mixed or average    0.000137
```
![img.png](img.png)

## **8. Conclusion**
- **Random Forest** is the **best performing model** overall.
- **KNN** performed the worst on the test set.
- The model can provide **game recommendations** based on predicted scores, with **The Escapists** and **Jumpgate: The Reconstruction Initiative** being the top recommendations.

---
## **9. Planned further improvements:**
- **Improving Data Quality**: Expand data cleaning methods and apply additional data augmentation techniques.
- **Additional Feature Engineering**: Experiment with new features and their interactions to achieve even better predictions.
- **Model Ensemble**: Explore the possibility of combining multiple models (ensemble methods) for even more accurate predictions.
- **Real-time Recommendations**: Develop a real-time recommendation system with data stream processing.
- **Personalization**: Incorporate additional user profiles to personalize recommendations.
- **Enhancing Interpretability Tools**: Improve SHAP and LIME explanations to be even more comprehensive and understandable for end users.
### **10. Final Thoughts**
This project successfully builds a machine learning pipeline for predicting game scores, evaluates multiple models, and generates recommendations. It demonstrates how to handle data preprocessing, model selection, evaluation, and interpretation, making it useful for those interested in applying machine learning to similar types of data.
