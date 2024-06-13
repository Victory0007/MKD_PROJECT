from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import cv2
import numpy as np
from easyocr import Reader
from similarity_search import similarity_search
import requests
import google.generativeai as genai

app = Flask(__name__)

# Load the ecommerce dataset
csv_file = 'Data/sampled_data.csv'
data = pd.read_csv(csv_file)
data.drop("Unnamed: 0", inplace=True, axis=1)


# Function to get natural language response from Gemini
# Configure Gemini API
genai.configure(api_key="YOUR-API-KEY")
model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(prompt):
    pro = (f"Please provide a one line friendly waiting response for this {prompt}, while my code searches the database for recommendations"
           f"Do not reply my prompt. Reply only the query.")
    response = model.generate_content(pro).text
    #return response['text'] if 'text' in response else None
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        gemini_response = get_gemini_response(query)
        response = similarity_search(query)
        matched_ids = [int(match['id'][3:]) for match in response['matches']]
        matched_products = data.loc[matched_ids].drop_duplicates(subset='StockCode')[["Description", "UnitPrice"]]
        return render_template('index.html', query=query, gemini_response=gemini_response,
                               products=matched_products.to_dict(orient='records'))
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
            print("EasyOCR Extracted: {}".format(output_txt))

            # Get Gemini response
            gemini_response = get_gemini_response(output_txt)

            # Perform similarity search
            response = similarity_search(output_txt)
            matched_ids = [int(match['id'][3:]) for match in response['matches']]
            matched_products = data.loc[matched_ids].drop_duplicates(subset='StockCode')[["Description", "UnitPrice"]]

            return render_template('image_query.html', query=output_txt, gemini_response=gemini_response,
                                   products=matched_products.to_dict(orient='records'))
    return render_template('image_query.html')


if __name__ == '__main__':
    app.run(debug=True)
