from flask import Flask, request, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import io

app = Flask(__name__)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embeddings(texts):
    embeddings = model.encode(texts)
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

        if 'Resume' not in df.columns or 'Filename' not in df.columns:
            return "The uploaded CSV file does not contain the required columns 'resume_text' and 'file_name'.", 400

        resumes = df['Resume'].tolist()
        file_names = df['Filename'].tolist()

        # Generate embeddings
        embeddings = generate_embeddings(resumes)

        # Perform clustering using Agglomerative Hierarchical Clustering
        num_clusters = 10  # You can make this dynamic based on user input
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        labels = clustering_model.fit_predict(embeddings)

        # Create a dictionary to map cluster labels to file names
        clusters = {}
        for label, file_name in zip(labels, file_names):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(file_name)

        return render_template('result.html', clusters=clusters)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
