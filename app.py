from flask import Flask, render_template, request, jsonify
import pandas as pd
from myfunctions import train_and_evaluate_xgboost, predict_popularity_score, get_top_words_correlated_with_popularity
from collections import Counter
from datetime import datetime
import sqlite3
import os

app = Flask(__name__)

db_path = 'my.db'

# З'єднання з базою даних
conn = sqlite3.connect(db_path)

# Зчитування таблиці з бази даних у датафрейм
combined_df = pd.read_sql('SELECT * FROM combined_df', conn)

# Закриття з'єднання
conn.close()

# Змінні для збереження результатів
train_result = None
predict_result = None
update_in_progress = False
last_update_time = None
xgb_model = None
tfidf_vectorizer = None
top_words = None 

# Тренування моделі при запуску сервера
train_result, xgb_model, tfidf_vectorizer = train_and_evaluate_xgboost(df=combined_df, text_column='text')

@app.route('/')
def index():
    global train_result, predict_result, update_in_progress, last_update_time
    return render_template('index.html', train_result=train_result, predict_result=predict_result, update_in_progress=update_in_progress, last_update_time=last_update_time)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Отримання тексту статті з форми
        user_input_text = request.form['user_input_text']

        # Передбачення популярності тексту
        predicted_result = predict_popularity_score(user_input_text, xgb_model, tfidf_vectorizer)

        # Повернення результату в шаблон
        return render_template('index.html', predict_result=predicted_result)

    # Викликається, якщо метод не є POST
    return render_template('index.html', predict_result=None)


@app.route('/show_top_words', methods=['GET', 'POST'])
def show_top_words():
    global top_words

    if request.method == 'POST':
        # Отримання слів
        top_words = get_top_words_correlated_with_popularity(df=combined_df, xgb_model=xgb_model, tfidf_vectorizer=tfidf_vectorizer)

    # Перевірка, чи top_words містить дані
    if top_words is None:
        return render_template('index.html', top_words=[])

    # Передача списку слів до шаблону
    return render_template('index.html', top_words=top_words)


if __name__ == '__main__':
    app.run(debug=True)
