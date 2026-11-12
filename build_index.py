import onnxruntime as ort
import numpy as np
import faiss
import pandas as pd
from datasets import load_dataset

session = ort.InferenceSession("visual_search.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
selected_data = dataset.to_pandas()
selected_data.drop(columns=['image']).to_csv("catalog_meta.csv", index=False)

embedding_dim = 512
index = faiss.IndexFlatIP(embedding_dim)

def preprocess(image_pil):
    img = image_pil.convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

batch_size = 128
embeddings_list = []

for i in range(0, len(dataset), batch_size):
    batch_images = dataset[i : i + batch_size]["image"]
    batch_tensors = np.stack([preprocess(img) for img in batch_images])
    batch_embeddings = session.run(None, {input_name: batch_tensors})[0]
    embeddings_list.append(batch_embeddings)

all_embeddings = np.vstack(embeddings_list)
index.add(all_embeddings)
faiss.write_index(index, "fashion_catalog.index")
