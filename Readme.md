
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

## **2. Feature Engineering and Data Preprocessing**

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

The dataset used contains reviews and sentiment data from critics and users, along with information about the platform and game titles. The key feature engineering steps include the creation of **ratio features** and the conversion of categorical sentiment data into usable inputs for machine learning models.

1. [Sentiment Imputation](#sentiment-imputation)
2. [Data Validation](#data-validation)

### Sentiment Imputation
The script handles missing values in several columns by applying fallback logic, which prioritizes filling in values from related columns and sets default values when necessary:

- **Critic Review Count**: Filled with `User Review Count`, or set to 0 if missing.
- **Critic Positive Count**: Filled with `User Positive Count`, or set to 0 if missing.
- **Critic Score**: Filled with `User Score * 10` if missing, or set to 0.
- **User Score**: Filled with `Critic Score / 10` if missing, or set to 0.

Additionally, the sentiment for both users and critics is imputed based on available data:
- **User Sentiment**: If user sentiment is unknown, it falls back to critic sentiment. If both are unknown, it uses the critic score to determine sentiment based on predefined thresholds:
  - `>= 90`: 'Universal acclaim'
  - `>= 75`: 'Generally favorable'
  - `>= 50`: 'Mixed or average'
  - `>= 20`: 'Generally unfavorable'
  - Otherwise: 'Overwhelming dislike'
  
- **Critic Sentiment**: If critic sentiment is unknown, it falls back to user sentiment. If both are unknown, it uses the user score to determine sentiment based on predefined thresholds:
  - `>= 9`: 'Universal acclaim'
  - `>= 7.5`: 'Generally favorable'
  - `>= 5`: 'Mixed or average'
  - `>= 2`: 'Generally unfavorable'
  - Otherwise: 'Overwhelming dislike'

### Data Validation
The script performs a few basic data quality checks to ensure data integrity:

1. **Missing Values**: It checks for missing values in the dataset before and after imputation, giving insights into which columns have missing data.
2. **Score Distribution**: The range of user and critic scores is printed to ensure there are no outlier values.
3. **Missing Sentiment**: The proportion of 'Unknown' sentiments is shown to check how well the imputation strategy worked.
4. **Column Information**: It strips any leading/trailing spaces in column names and checks the data types of each column.

### Sample Output:
- Updated sentiment distribution with counts for user and critic sentiment.
- Data quality reports showing missing values before and after processing, score distributions, and sentiment analysis.

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
- Processed shape: (35413, 12)
- X_train shape: (28330, 12) X_test shape: (7083, 12)


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
Hyperparameter grids for **GridSearchCV**: for three different machine learning models: **Random Forest**, **Neural Network (MLP)**, and **K-Nearest Neighbors (KNN)**. The goal is to find the best set of hyperparameters for each model that optimizes performance on the training dataset.

```python
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
    'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}
}
```

### 1. **Random Forest Regressor** (`RandomForestRegressor`):
The following hyperparameters are tuned:
- `n_estimators`: The number of trees in the forest. We are trying values `[100, 200]`.
- `max_depth`: The maximum depth of the tree. We test both `None` (no limit) and a depth of `10`.

```python
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
}
```

- **Model Used**: `RandomForestRegressor`
- **Cross-validation**: 5-fold cross-validation
- **Best Parameters**: The best parameters are selected based on the grid search results and are printed to the console.

### 2. **Neural Network** (`MLPRegressor`):
The following hyperparameters are tuned:
- `hidden_layer_sizes`: The number of neurons in each hidden layer. We try two different configurations: `(50,)` and `(100,)`.
- `alpha`: L2 regularization term. We test values `[0.0001, 0.001]`.

```python
param_grids = {
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
}
```

- **Model Used**: `MLPRegressor` (Multi-layer Perceptron Regressor)
- **Cross-validation**: 5-fold cross-validation
- **Best Parameters**: The best parameters are selected based on the grid search results and are printed to the console.

### 3. **K-Nearest Neighbors** (`KNeighborsRegressor`):
The following hyperparameters are tuned:
- `n_neighbors`: The number of neighbors to use for prediction. We test values `[3, 5]`.
- `weights`: The weighting function used in prediction. We try both `'uniform'` (all neighbors have equal weight) and `'distance'` (closer neighbors have higher weight).

```python
param_grids = {
    'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},
}
```

- **Model Used**: `KNeighborsRegressor`
- **Cross-validation**: 5-fold cross-validation
- **Best Parameters**: The best parameters are selected based on the grid search results and are printed to the console.

### **Model Training and Evaluation:**
For each model:
- The **best parameters** were selected using **GridSearchCV**.
- **Model performance** was evaluated on both **cross-validation** and **test set** using metrics like **RMSE**, **MAE**, **R²**, and **NDCG@10**.

## **4. Evaluation Results**
The **final results** after evaluation on the test set showed the following:

- **Random Forest** was the best-performing model with the lowest **RMSE** (0.456) and highest **R²** (0.978).
- **KNN** performed the worst on the test set with a **R²** of 0.954 and **RMSE** of 0.661.

```python

Final Model Comparison (Test Set):
            Model  RMSE (Test)  MAE (Test)  R² (Test)   NDCG@10
0   Random Forest     0.461958    0.286607   0.977579  0.991595
1  Neural Network     0.464687    0.300675   0.977313  0.985284
2             KNN     0.494412    0.306819   0.974318  0.985076
3     Transformer     0.496454    0.325225   0.974106  0.988830
```
### Predictive Accuracy (RMSE & MAE)

- **Random Forest** has the lowest RMSE (0.461958) and lowest MAE (0.286607), indicating it makes the smallest average prediction errors.
- **Neural Network** follows closely but lags slightly in both metrics.
- **KNN** and **Transformer** perform worse, with Transformer having the highest errors (RMSE: 0.496, MAE: 0.325).

### Explained Variance (R²)

- **Random Forest** again leads with R² = 0.9776, meaning it explains ~97.8% of the variance in the target variable.
- Other models trail marginally (**Neural Network**: 0.9773) or more noticeably (**Transformer**: 0.9741).

### Ranking Quality (NDCG@10)

- **Random Forest** achieves the highest NDCG@10 (0.9916), showing superior ranking of top recommendations.
- **Transformer** surprisingly outperforms **Neural Network** and **KNN** in ranking (NDCG@10 = 0.9888 vs. 0.9852/0.9851), despite weaker RMSE/MAE. This suggests it captures item relevance order better, even if absolute rating predictions are less accurate.

### Key Insights

- **Random Forest** is the best overall model across all metrics. It balances high predictive accuracy (lowest RMSE/MAE), strong explanatory power (highest R²), and excellent ranking performance (best NDCG@10).
- **Transformer** has decent ranking capability (2nd-best NDCG@10) but struggles with precise numerical predictions, making it less suitable for applications requiring exact rating estimates.
- **Neural Network** is a close second to **Random Forest** in accuracy but lags in ranking, while **KNN** is the weakest overall.
## **5. Model Recommendations**
### **Top Game Recommendations:**
The model's **top 5 recommendations** were based on the predicted scores for each game:
```python

Top Recommendations:
                                         Title        Platform  Predicted_Score
16248                            The Escapists  ios-iphoneipad         9.604445
9754   Jumpgate: The Reconstruction Initiative              PC         9.604445
14719                          Visions of Mana   Xbox Series X         9.586389
34042                   Cooking Mama: Cookstar   PlayStation 4         9.582155
4621            Metro Exodus: Enhanced Edition        Xbox One         9.582155
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

### **11. Requirments**

 ```bash
pip install -r requirements.txt
 ```