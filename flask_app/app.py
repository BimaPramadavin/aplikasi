from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Preprocessing steps (as defined earlier)
    numerical_features = ['Jumlah Barang', 'Harga Satuan', 'Jumlah']
    categorical_features = ['Size', 'Kategori']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    data_preprocessed = preprocessor.fit_transform(data)
    return pd.DataFrame(data_preprocessed, columns=numerical_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)))

def perform_clustering(data_preprocessed_df):
    clustering = AgglomerativeClustering(n_clusters=2)
    data_preprocessed_df['Cluster'] = clustering.fit_predict(data_preprocessed_df)
    return data_preprocessed_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess data
            data_preprocessed_df = preprocess_data(file_path)

            # Perform clustering
            clustered_data = perform_clustering(data_preprocessed_df)

            # Save clustered data to CSV
            clustered_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clustered_data.csv')
            clustered_data.to_csv(clustered_file_path, index=False)

            # Plotting and saving the visualization
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=clustered_data['Jumlah Barang'], y=clustered_data['Harga Satuan'], hue=clustered_data['Cluster'], palette='viridis')
            plt.title('Visualisasi Kluster Berdasarkan Jumlah Barang dan Harga Satuan')
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'cluster_plot.png'))

            return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html', plot_url='uploads/cluster_plot.png', download_url='uploads/clustered_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
