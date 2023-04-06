from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
from flask import Flask, render_template, request, redirect, url_for

# Load the BERT model and tokenizer
path = "D:/2/dataset/BERT_sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)

# Initialize the Flask app
app = Flask(__name__)

# Define the base template
@app.route("/")
def home():
    return render_template("base.html")

# Define the route for the prediction form
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input sentence from the form
    sentence = request.form["sentence"]
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    sentiment_labels = ['negative', 'neutral', 'positive']
    # Map the predicted label to the sentiment label
    sentiment = sentiment_labels[predicted_label]

    # Redirect to the result route with the predicted sentiment label as a parameter
    return redirect(url_for("result", prediction=sentiment))

# Define the route for the prediction result
@app.route("/result")
def result():
    # Get the predicted sentiment label from the parameter
    sentiment = request.args.get("prediction")

    # Render the result.html template with the prediction result
    return render_template("result.html", prediction=sentiment)

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
