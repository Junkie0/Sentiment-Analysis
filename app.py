from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pre-trained models and vectorizer
with open('Model2/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Model2/logistic_regression.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('Model2/linear_svm.pkl', 'rb') as f:
    linear_svm_model = pickle.load(f)

with open('Model2/multinomial_naive_bayes.pkl', 'rb') as f:
    multinomial_nb_model = pickle.load(f)

models = {
    'Logistic Regression': logistic_regression_model,
    'Linear SVM': linear_svm_model,
    'Multinomial Naive Bayes': multinomial_nb_model
}


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    selected_model_name = "Logistic Regression"  # default model

    if request.method == 'POST':
        try:
            # Get the input text and selected model
            input_text = request.form['input_text']
            selected_model_name = request.form['model']

            # Preprocess the input text
            input_vector = vectorizer.transform([input_text])

            # Predict using the selected model
            selected_model = models[selected_model_name]
            prediction = selected_model.predict(input_vector)

            # Convert prediction to sentiment
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
        except Exception as e:
            print(f"Error processing request: {e}")
            sentiment = "Error processing request"

    return render_template('index.html',
                         sentiment=sentiment,
                         selected_model=selected_model_name,
                         models=list(models.keys()))

if __name__ == '__main__':
    app.run(debug=False)
