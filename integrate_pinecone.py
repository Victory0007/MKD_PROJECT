"""Create product embeddings and upsert them into a Pinecone index."""
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_INDEX_NAME = "mkd-ecommerce-ds-project"


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value


def load_products(csv_path, max_per_description):
    data = pd.read_csv(csv_path)
    required_columns = {"Description", "StockCode", "UnitPrice"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    data = data.dropna(subset=["Description"]).copy()
    data["Description"] = data["Description"].astype(str).str.strip()
    data = data[data["Description"] != ""]
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data = data.drop_duplicates(subset=["StockCode", "Description", "UnitPrice"])

    if max_per_description > 0:
        data = (
            data.groupby("Description", group_keys=False)
            .apply(lambda group: group.sample(
                n=min(len(group), max_per_description),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
    else:
        data = data.reset_index(drop=True)
    return data


def ensure_index(client, index_name, dimension, cloud, region):
    existing = client.list_indexes().names()
    if index_name not in existing:
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/Cleaned_Ecommerce_dataset.csv")
    parser.add_argument("--max-per-description", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero.")
    if args.max_per_description < 0:
        parser.error("--max-per-description cannot be negative.")

    products = load_products(args.input, args.max_per_description)
    if products.empty:
        raise RuntimeError("No usable product records were found in the dataset.")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        products["Description"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    client = Pinecone(api_key=required_env("PINECONE_API_KEY"))
    ensure_index(
        client,
        index_name,
        dimension=int(embeddings.shape[1]),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
    )
    index = client.Index(index_name)

    for start in range(0, len(products), args.batch_size):
        end = min(start + args.batch_size, len(products))
        vectors = []
        for position in range(start, end):
            row = products.iloc[position]
            metadata = {
                "description": str(row["Description"]),
                "stock_code": str(row["StockCode"]),
            }
            if pd.notna(row["UnitPrice"]):
                metadata["unit_price"] = float(row["UnitPrice"])
            vectors.append((
                f"product-{position}",
                embeddings[position].tolist(),
                metadata,
            ))
        index.upsert(vectors=vectors)
        print(f"Uploaded {end:,}/{len(products):,} products")

    output_path = Path("data") / "sampled_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    products.to_csv(output_path, index=False)
    print(f"Saved the indexed product sample to {output_path}")


if __name__ == "__main__":
    main()
