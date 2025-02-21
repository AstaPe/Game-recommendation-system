# Suppress TensorFlow log messages (info-level and below)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info-level logs from TensorFlow

# Import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, ndcg_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import shap
import lime.lime_tabular
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input, Reshape
from tensorflow.keras.optimizers import Adam

# ========== README ==========
"""
Recommendation System for Video Games

Overview:
This project is a video game recommendation system that predicts user scores based on historical review data.
It uses multiple machine learning regression models including Random Forests, Neural Networks (MLP), K-Nearest Neighbors, 
and a Transformer-based neural network to generate predictions. The system also provides interpretability via SHAP and LIME.

Inputs:
- Raw game data CSV with columns:
  - Title: The name of the video game.
  - Platform: The gaming platform (e.g., PC, Console).
  - Release_Date: The release date of the game.
  - Critic_Score: Average score provided by critics.
  - User_Score: Average score provided by users (target variable).
  - Critic_Sentiment: Sentiment classification (e.g., Universal acclaim, Mixed or average) based on critic reviews.
  - User_Sentiment: Sentiment classification based on user reviews.
  - Critic_Review_Count: Number of reviews from critics.
  - User_Review_Count: Number of reviews from users.
  - Critic_Positive, Critic_Neutral, Critic_Negative: Breakdown of critic review opinions.
  - User_Positive, User_Neutral, User_Negative: Breakdown of user review opinions.

Outputs:
- Trained recommendation models.
- Evaluation metrics including RMSE, MAE, R², and NDCG@10.
- A list of top game recommendations based on predicted user scores.
- Explanations of model predictions using SHAP (global feature importance) and LIME (local instance explanations).

Functions Explanation:
- map_sentiment(row):
    - Purpose: Impute missing or unknown user sentiment by using critic sentiment or converting critic scores to a sentiment label.
    - Input: A single row of the dataset.
    - Output: A sentiment label (e.g., 'Universal acclaim', 'Mixed or average').
- build_transformer_model(input_shape):
    - Purpose: Build a transformer-based neural network for regression tasks.
    - Input: The number of features in the processed input data.
    - Output: A compiled Keras model that outputs a single continuous value (User Score).
- recommend_games(model, n=5, diversity=0.1):
    - Purpose: Generate top N game recommendations from the dataset using the trained model.
    - Input: A trained model, the desired number of recommendations, and a diversity factor.
    - Output: A DataFrame with recommended game titles, their platforms, and predicted scores.

Workflow and Expected Results:
1. Data Preparation: Handle missing values, normalize numerical features, and encode categorical variables.
2. Feature Engineering: Create ratio features (e.g., approval ratios) to capture relationships between positive reviews and total reviews.
3. Model Training: Train multiple regression models with hyperparameter tuning using GridSearchCV and cross-validation.
4. Evaluation: Assess models using RMSE, MAE, R², and NDCG@10. For example, a Random Forest might achieve:
   - Test RMSE: ~0.455
   - Test MAE: ~0.277
   - Test R²: ~0.978
   - NDCG@10: ~0.989
5. Recommendations: Produce a ranked list of recommended games based on predicted user scores.
6. Interpretability: Use SHAP for global feature importance and LIME for local explanations to understand model decisions.

Aims:
- To build an effective recommendation system that accurately predicts video game user scores.
- To provide interpretable insights into model decisions via SHAP and LIME.
- To assist stakeholders in making data-driven decisions regarding game recommendations.

Future Improvements:
- Enhance data cleaning and augmentation techniques.
- Explore additional feature engineering methods and interactions.
- Experiment with model ensembling to further improve prediction accuracy.
- Develop a real-time recommendation engine.
- Incorporate user personalization to tailor recommendations further.

Conclusion:
This system demonstrates a robust approach to predicting video game user scores and generating recommendations, combining advanced machine learning techniques with interpretability tools.
"""

# ========== DATA LOADING ==========
try:
    df = pd.read_csv(r'C:\Users\astap\OneDrive\Documents\Sprendimai\Baigiamasis\updated_game_reviews.csv',
                     parse_dates=['Release Date'])
    print("Data loaded successfully. Columns:", df.columns)
except Exception as e:
    print("Error loading data:", e)
    exit()

# ========== SENTIMENT IMPUTATION ==========
# Fill missing values using fallback logic
df['Critic Review Count'] = df['Critic Review Count'].fillna(df['User Review Count']).fillna(0)
df['Critic Positive Count'] = df['Critic Positive Count'].fillna(df['User Positive Count']).fillna(0)
df['Critic Score'] = df['Critic Score'].fillna(df['User Score'] * 10).fillna(0)
df['User Score'] = df['User Score'].fillna(df['Critic Score'] / 10).fillna(0)

def map_sentiment(row):
    """Handle Unknown user sentiment using critic data"""
    if row['User Sentiment'] != 'Unknown':
        return row['User Sentiment']
    if row['Critic Sentiment'] != 'Unknown':
        return row['Critic Sentiment']
    if not pd.isna(row['Critic Score']):
        normalized_score = row['Critic Score']
        if normalized_score >= 90:
            return 'Universal acclaim'
        if normalized_score >= 75:
            return 'Generally favorable'
        if normalized_score >= 50:
            return 'Mixed or average'
        if normalized_score >= 20:
            return 'Generally unfavorable'
        return 'Overwhelming dislike'
    return 'Overwhelming dislike'

def map_critic_sentiment(row):
    """Handle Unknown critic sentiment using user sentiment and user score"""
    if row['Critic Sentiment'] != 'Unknown':
        return row['Critic Sentiment']
    if row['User Sentiment'] != 'Unknown':
        return row['User Sentiment']
    if not pd.isna(row['User Score']):
        if row['User Score'] >= 9:
            return 'Universal acclaim'
        if row['User Score'] >= 7.5:
            return 'Generally favorable'
        if row['User Score'] >= 5:
            return 'Mixed or average'
        if row['User Score'] >= 2:
            return 'Generally unfavorable'
        return 'Overwhelming dislike'
    return 'Overwhelming dislike'


# Apply sentiment mapping
df['User Sentiment'] = df.apply(map_sentiment, axis=1).fillna('Overwhelming dislike')
df['Critic Sentiment'] = df.apply(map_critic_sentiment, axis=1).fillna('Overwhelming dislike')

print("\nUpdated Sentiment Distribution:")
print(df['User Sentiment'].value_counts())
print(df['Critic Sentiment'].value_counts())

# ========== DATA VALIDATION ==========
print("\nData Quality Checks:")
print("Missing Values Before Processing:")
print(df.isna().sum())

print("\nScore Distributions:")
print(f"User Score Range: {df['User Score'].min()} - {df['User Score'].max()}")
print(f"Critic Score Range: {df['Critic Score'].min()} - {df['Critic Score'].max()}")

print("\nMissing Sentiment Analysis:")
print(df['User Sentiment'].value_counts(normalize=True))

# Strip leading/trailing spaces in column names
df.columns = df.columns.str.strip()
print("Data types of columns:")
print(df.dtypes)

# Replace 'N/A' with 0 and pd.NA with np.nan
df.replace({'N/A': 0, pd.NA: np.nan}, inplace=True)

# ========== FEATURE ENGINEERING ==========
try:
    df['Release_Year'] = df['Release Date'].dt.year
    df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

    # Create ratio features (only these will be used for training)
    df['Critic_Approval_Ratio'] = np.where(
        df['Critic Review Count'] == 0, 0,
        df['Critic Positive Count'] / df['Critic Review Count']
    )
    df['User_Approval_Ratio'] = np.where(
        df['User Review Count'] == 0, 0,
        df['User Positive Count'] / df['User Review Count']
    )
    print("Feature engineering completed.")
except Exception as e:
    print("Error during feature engineering:", e)
    exit()

# For model training we use only ratio features and selected categorical sentiment features.
numerical_cols = ['Critic_Approval_Ratio', 'User_Approval_Ratio']
categorical_cols = ['Critic Sentiment', 'User Sentiment']  # 'Platform' is removed for training

# Fill missing values for these columns
df[numerical_cols] = df[numerical_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

print("Missing values after filling:")
print(df.isnull().sum())
print("First few rows of the DataFrame:")
print(df.head())

# ========== MODEL SETUP ==========
# Use only the ratio features and the two sentiment columns for training.
try:
    X = df[numerical_cols + categorical_cols]
    y = df['User Score'].fillna(df['User Score'].median())
    print("Features (X) and target (y) created.")
except Exception as e:
    print("Error creating features and target:", e)
    exit()

# Ensure categorical columns are of type 'object'
for col in categorical_cols:
    if X[col].dtype != 'object':
        print(f"Warning: Column '{col}' is not of type 'object'. Converting to object.")
        X[col] = X[col].astype('object')

print("Columns in X:")
print(X.columns)
print("Missing values in X:")
print(X.isnull().sum())

# ========== PREPROCESSING ==========
# To remove the "Unknown" category for User Sentiment during one-hot encoding,
# we manually specify the categories. For "Critic Sentiment" we use all its unique values,
# and for "User Sentiment" we exclude "Unknown".
critic_sentiment_categories = sorted(df['Critic Sentiment'].unique())
user_sentiment_categories = sorted([cat for cat in df['User Sentiment'].unique() if cat != 'Unknown'])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False,
                                     categories=[critic_sentiment_categories, user_sentiment_categories]))
        ]), categorical_cols)
    ]
)

try:
    X_processed = preprocessor.fit_transform(X)
    print("Data preprocessing completed. Processed shape:", X_processed.shape)
except Exception as e:
    print("Error during preprocessing:", e)
    exit()

# ========== TRAIN-TEST SPLIT ==========
try:
    tscv = TimeSeriesSplit(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, shuffle=False)
    print("Train-test split completed. X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
except Exception as e:
    print("Error during train-test split:", e)
    exit()

# ========== MODEL TRAINING ==========
models = {
    'Random Forest': RandomForestRegressor(),
    'Neural Network': MLPRegressor(max_iter=1000),
    'KNN': KNeighborsRegressor(),
    'Transformer': None  # Will be defined separately
}

# Hyperparameter grids for GridSearchCV
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
    'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}
}

# Random Forest hiperparametrų paieška
rf = RandomForestRegressor()
rf_grid = param_grids['Random Forest']
rf_search = GridSearchCV(rf, rf_grid, cv=5)
rf_search.fit(X_train, y_train)
print("Best Random Forest Params:", rf_search.best_params_)

# Neural Network hiperparametrų paieška
nn = MLPRegressor(max_iter=1000)
nn_grid = param_grids['Neural Network']
nn_search = GridSearchCV(nn, nn_grid, cv=5)
nn_search.fit(X_train, y_train)
print("Best Neural Network Params:", nn_search.best_params_)

# KNN hiperparametrų paieška
knn = KNeighborsRegressor()
knn_grid = param_grids['KNN']
knn_search = GridSearchCV(knn, knn_grid, cv=5)
knn_search.fit(X_train, y_train)
print("Best KNN Params:", knn_search.best_params_)


def build_transformer_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    # Reshape to 3D for MultiHeadAttention
    x = Reshape((1, 64))(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

transformer_model = build_transformer_model(X_train.shape[1])
models['Transformer'] = transformer_model

best_models = {}
cv_results = []

for name, model in models.items():
    if name == 'Transformer':
        history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                            validation_split=0.2, verbose=0)
        best_models[name] = model
        cv_results.append({
            'Model': name,
            'Best Params': 'N/A',
            'RMSE (CV)': np.sqrt(min(history.history['val_loss'])),
            'R² (CV)': r2_score(y_train, model.predict(X_train))
        })
    else:
        grid = GridSearchCV(model, param_grids[name], cv=tscv, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        cv_results.append({
            'Model': name,
            'Best Params': grid.best_params_,
            'RMSE (CV)': np.sqrt(-grid.best_score_),
            'R² (CV)': grid.best_estimator_.score(X_train, y_train)
        })

# ========== EVALUATION ==========
final_results = []
for name, model in best_models.items():
    if name == 'Transformer':
        y_pred = model.predict(X_test).flatten()
    else:
        y_pred = model.predict(X_test)
    # Reshape for ndcg_score
    y_test_2d = y_test.values.reshape(1, -1)
    y_pred_2d = y_pred.reshape(1, -1)
    final_results.append({
        'Model': name,
        'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE (Test)': mean_absolute_error(y_test, y_pred),
        'R² (Test)': r2_score(y_test, y_pred),
        'NDCG@10': ndcg_score(y_test_2d, y_pred_2d, k=10)
    })

# ========== FEATURE IMPORTANCE ==========
if 'Random Forest' in best_models:
    rf_model = best_models['Random Forest']
    feature_importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance (Random Forest):")
    print(importance_df.head(10))

    print("\n=== Feature Validation ===")
    # Check correlation among the numerical (ratio) features
    corr_matrix = df[numerical_cols].corr().abs()
    high_corr = corr_matrix[corr_matrix > 0.8].stack().reset_index()
    high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
    print("\nHigh Correlations (>0.8):")
    print(high_corr.drop_duplicates())

    print("\nLow Importance Features:")
    low_importance = importance_df[importance_df['Importance'] < 0.01]
    print(low_importance)

# ========== CROSS-VALIDATION RESULTS ==========
print("\nCross-Validation and Hyperparameter Tuning Results:")
cv_results_df = pd.DataFrame(cv_results)
print(cv_results_df)

# ========== FINAL MODEL COMPARISON ==========
print("\nFinal Model Comparison (Test Set):")
final_results_df = pd.DataFrame(final_results)
print(final_results_df)

# ========== RECOMMENDATION ==========
def recommend_games(model, n=5, diversity=0.1):
    # Check if 'Platform' column exists in the dataframe
    if 'Platform' not in df.columns:
        raise ValueError("The 'Platform' column is missing in the dataframe.")

    # Generate predictions on the full processed dataset.
    df['Predicted_Score'] = model.predict(X_processed)

    # Sort values based on the predicted score
    df_sorted = df.sort_values('Predicted_Score', ascending=False)

    # Apply platform diversity filter (add diversity within the top recommendations)
    recommendations = (
        df_sorted.groupby('Platform')
        .head(int(n * (1 + diversity)))  # Get top 'n' games with diversity
        .head(n)  # Select top 'n' after considering diversity
    )

    # Ensure 'Platform' column is included in the final recommendations
    return recommendations[['Title', 'Platform', 'Predicted_Score']]

# Example: Get recommendations from the best Random Forest model.
best_model = best_models['Random Forest']
recommendations = recommend_games(best_model)
print("\nTop Recommendations:")
print(recommendations)

# ========== SHAP EXPLANATIONS ==========
explainer = shap.TreeExplainer(best_models['Random Forest'])
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, feature_names=preprocessor.get_feature_names_out())

# ========== LIME EXPLANATIONS ==========
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=['User_Score'],
    mode='regression'
)
exp = lime_explainer.explain_instance(X_test[0], best_models['Random Forest'].predict, num_features=10)
exp.show_in_notebook()

