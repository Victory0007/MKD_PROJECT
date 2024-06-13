# import pandas as pd
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import math
# # Load the ecommerce dataset
# csv_file = 'sampled_data.csv'
# data = pd.read_csv(csv_file)
# data.drop("Unnamed: 0", inplace=True, axis=1)
#
# # Define a function to create embeddings using OpenAI
# # Load the Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# # Initialize Pinecone
# pinecone_api_key = "bba7a9c0-eb9c-4397-9b07-2019752f54ea"
# pc = Pinecone(api_key=pinecone_api_key)
#
# index_name = "mkd-ecommerce-ds-project"
#
# def similarity_search(index, query, top_k=5):
#     # Encode the query using the same Sentence Transformer model
#     query_embedding = model.encode([query]).tolist()
#     # Perform the search in Pinecone
#     results = index.query(vector=query_embedding, top_k=top_k, include_values=False)
#     return results
#
# index = pc.Index(index_name)
#
# # Example usage of similarity search
# query = "I am looking for pink bags"
# response = similarity_search(index, query)
#
# # Retrieve full product details for the matches
# matched_ids = [int(match['id'][3:]) for match in response['matches']]
# print(len(matched_ids))
# matched_products = data.loc[matched_ids].drop_duplicates(subset='StockCode')[["Description", "UnitPrice"]]
#
# # Print the search results
# print(matched_products)

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone and Sentence Transformer model
pinecone_api_key = "bba7a9c0-eb9c-4397-9b07-2019752f54ea"
pc = Pinecone(api_key=pinecone_api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_search(query, top_k=5):
    # Encode the query using the Sentence Transformer model
    query_embedding = model.encode([query]).tolist()
    # Perform the search in Pinecone
    index_name = "mkd-ecommerce-ds-project"
    index = pc.Index(index_name)
    results = index.query(vector=query_embedding, top_k=top_k, include_values=False)
    return results
