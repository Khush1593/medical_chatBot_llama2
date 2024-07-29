from src.helper import load_data, download_hugging_face_embedding, text_splitter
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()
from transformers import AutoModel, AutoTokenizer
import torch

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_data("data/")
text_chunk = text_splitter(extracted_data)
embedding = download_hugging_face_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY, region=PINECONE_API_ENV)

# Load Hugging Face model

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
index = pc.Index("mchatbot")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Create embeddings for each chunk
embeddings = [get_embedding(chunk.page_content) for chunk in text_chunk]

# Prepare data for upserting into Pinecone
pinecone_data = [
    {
        "id": str(i),
        "values": embedding,
        "metadata": {"text": chunk.page_content}
    }
    for i, (chunk, embedding) in enumerate(zip(text_chunk, embeddings))
]

print("Starts to load data into pinecode Database.....")
# Upsert the embeddings into Pinecone
batch_size = 100  # Adjust the batch size as needed to fit within the 4 MB limit
for i in range(0, len(pinecone_data), batch_size):
    batch = pinecone_data[i:i + batch_size]
    index.upsert(vectors=batch)

print("Data is loaded successfully in Database!")
 