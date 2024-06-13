import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from collections import Counter

# Load the ecommerce dataset
csv_file = 'Data/Cleaned_Ecommerce_dataset.csv'
data = pd.read_csv(csv_file)
data.drop("Unnamed: 0", inplace=True, axis=1)

# Sample input data to ensure equal representation of each description category
description_counts = Counter(data['Description'])
min_count = min(description_counts.values())
sampled_data = data.groupby('Description').apply(lambda x: x.sample(min(200000 // len(description_counts), len(x))))
sampled_data = sampled_data.reset_index(drop=True)

# Define a function to create embeddings using OpenAI
# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the descriptions
embeddings = model.encode(sampled_data['Description'].tolist())

# Initialize Pinecone
pinecone_api_key = "bba7a9c0-eb9c-4397-9b07-2019752f54ea"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "mkd-ecommerce-ds-project"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Use StockID as unique ID
# Generate unique IDs for each vector
ids = [f'id_{i}' for i in range(len(embeddings))]

index = pc.Index(index_name)

# Upsert embeddings to Pinecone
def upsert_vectors(index, vectors, ids):
    existing_ids = index.list(namespace='default')  # Fetch existing IDs
    new_vectors = [(id, vector) for id, vector in zip(ids, vectors) if id not in existing_ids]
    if new_vectors:
        index.upsert(vectors=new_vectors)

batch_size = 100
for i in range(0, len(embeddings), batch_size):
    i_end = min(i + batch_size, len(embeddings))
    batch_vectors = embeddings[i:i_end]
    batch_ids = ids[i:i_end]
    upsert_vectors(index, batch_vectors.tolist(), batch_ids)