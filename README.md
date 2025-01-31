# Sentiment Analysis Web App 🎭

This is a Flask-based web application for sentiment analysis. It allows users to input text and analyze its sentiment using different machine learning models.

![SS](./Images/Screenshot%202025-01-31%20095405.png)

## 🚀 Features

- Three sentiment analysis models:
  - Logistic Regression
  - Linear Support Vector Machine (SVM)
  - Multinomial Naive Bayes
- User-friendly web interface built with Flask and HTML/CSS.
- Text vectorization using TF-IDF.
- Displays sentiment as **Positive** or **Negative**.

## 🛠 Installation

### 1️⃣ Set Up a Virtual Environment

```sh
    python -m venv senti
    source senti/bin/activate  # On Windows use `senti\Scripts\activate`
```

### 2️⃣ Install Dependencies

```sh
  pip install pandas matplotlib seaborn nltk wordcloud scikit-learn plotly datasets flask pickle-mixin
```

### 3️⃣ Run the Application

```sh
  python app.py
```

Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 📂 Project Structure

```
📦 Sentiment Analysis
 ┣ 📂 Models
 ┃ ┣ 📜 linear_svm.pkl
 ┃ ┣ 📜 logistic_regression.pkl
 ┃ ┣ 📜 multinomial_naive_bayes.pkl 
 ┃ ┗ 📜 tfidf_vectorizer.pkl 
 ┣ 📂 static
 ┃ ┗ 📜 style.css
 ┣ 📂 templates
 ┃ ┗ 📜 index.html
 ┣ 📜 app.py
 ┗ 📜 README.md
```

## 📝 Notes

- Ensure the `Models` directory contains the pre-trained model files (`.pkl`).
- You can extend this project by adding more models or an improved UI.
