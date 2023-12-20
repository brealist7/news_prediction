import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from collections import Counter

def train_and_evaluate_xgboost(df, text_column='lemmatized_text_mystem', target_column='popularity_score', max_features=1000, test_size=0.2, random_state=42):
    # Використання TF-IDF для отримання числових ознак з текстових даних
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)  
    X_tfidf = tfidf_vectorizer.fit_transform(df[text_column]).toarray()

    # Додавання до ознак TF-IDF числових даних
    X = pd.DataFrame(X_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

    # Отримання цільової ознаки (popularity_score)
    y = df[target_column]

    # Розділення даних на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Модель градієнтного бустингу (XGBoost)
    xgb_model = XGBRegressor(random_state=random_state)
    xgb_model.fit(X_train, y_train)

    # Оцінка результатів для градієнтного бустингу
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Повернення значень
    return {
        'model': 'XGBoost',
        'data': text_column,
        'mse': mse_xgb,
        'r2': r2_xgb
    }, xgb_model, tfidf_vectorizer

def predict_popularity_score(text, xgb_model, tfidf_vectorizer):
    # Векторизація введеного тексту за допомогою того самого TF-IDF векторизатора
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()

    # Створення DataFrame з числовими ознаками TF-IDF
    text_df = pd.DataFrame(text_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

    # Прогноз popularity_score за допомогою навченої моделі
    predicted_score = xgb_model.predict(text_df)

    return predicted_score[0]

def get_top_words_correlated_with_popularity(df, xgb_model, tfidf_vectorizer, text_column='lemmatized', max_features=1000, random_state=42):
    # Використайте TF-IDF для отримання числових ознак з текстових даних
    X_tfidf = tfidf_vectorizer.transform(df[text_column]).toarray()

    # Додаваня до ознак TF-IDF числових даних
    X = pd.DataFrame(X_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

    # Одержання цільової ознаки (popularity_score)
    y = df['popularity_score']

    # Модель градієнтного бустингу (XGBoost)
    xgb_model.fit(X, y)

    # Отримання важливості ознак
    feature_importances = xgb_model.feature_importances_

    # Знаходження індексів та назв слів, які найбільше корелюють із успішністю тексту
    top_feature_indices = feature_importances.argsort()[::-1][:max_features]
    top_words = [tfidf_vectorizer.get_feature_names_out()[idx] for idx in top_feature_indices if feature_importances[idx] > 0]

    # Повернення списку слів
    return top_words
