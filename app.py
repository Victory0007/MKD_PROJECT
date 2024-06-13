from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import cv2
from easyocr import Reader
from similarity_search import similarity_search
import numpy as np

app = Flask(__name__)

# Load the ecommerce dataset
csv_file = 'sampled_data.csv'
data = pd.read_csv(csv_file)
data.drop("Unnamed: 0", inplace=True, axis=1)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        response = similarity_search(query)
        matched_ids = [int(match['id'][3:]) for match in response['matches']]
        matched_products = data.loc[matched_ids].drop_duplicates(subset='StockCode')[["Description", "UnitPrice"]]
        return render_template('index.html', query=query, products=matched_products.to_dict(orient='records'))
    return render_template('index.html')


@app.route('/image_query', methods=['GET', 'POST'])
def image_query():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the image
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # Initialize EasyOCR reader
            reader = Reader(['en'])

            # Perform OCR using EasyOCR
            result = reader.readtext(image)

            # Extracted text
            output_txt = " ".join([text[1] for text in result])
            # print("EasyOCR Extracted: {}".format(output_txt))

            # Perform similarity search
            response = similarity_search(output_txt)
            matched_ids = [int(match['id'][3:]) for match in response['matches']]
            matched_products = data.loc[matched_ids].drop_duplicates(subset='StockCode')[["Description", "UnitPrice"]]

            return render_template('image_query.html', query=output_txt,
                                   products=matched_products.to_dict(orient='records'))
    return render_template('image_query.html')


if __name__ == '__main__':
    app.run(debug=True)
