from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import KMeans
import io

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded.", 400
        
        # Read the CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)
        resumes = df['resume_text'].tolist()

        # Generate embeddings
        embeddings = generate_embeddings(resumes)

        # Perform clustering
        num_clusters = 5  # You can make this dynamic based on user input
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        # Add cluster labels to the dataframe
        df['Cluster'] = labels

        # Convert dataframe to HTML table
        return render_template('result.html', tables=[df.to_html(classes='data', header="true")])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
