import io
import numpy as np
import pandas as pd
import faiss
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Visual Search API", description="Движок визуального поиска одежды")

# При старте сервиса один раз грузим в память веса, индекс и метаданные
print("Загрузка ONNX модели...")
session = ort.InferenceSession("visual_search.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

print("Загрузка FAISS индекса и метаданных...")
index = faiss.read_index("fashion_catalog.index")
df_meta = pd.read_csv("catalog_meta.csv")

def preprocess(image_bytes):
    # Конвертируем байты обратно в PIL Image и делаем тот же препроцессинг
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # Добавляем ось батча, т.к. ONNX ждет тензор (1, 3, 224, 224)
    return np.expand_dims(img_np, axis=0)

@app.post("/search")
async def search_similar(file: UploadFile = File(...), top_k: int = 5):
    # 1. Читаем загруженный файл
    image_bytes = await file.read()
    tensor = preprocess(image_bytes)
    
    # 2. Инференс (получаем вектор 512)
    embedding = session.run(None, {input_name: tensor})[0]
    
    # 3. Векторный поиск (L2/Cosine)
    distances, indices = index.search(embedding, top_k)
    
    # 4. Формируем ответ, подтягивая названия из датафрейма
    results = []
    for i, idx in enumerate(indices[0]):
        item_meta = df_meta.iloc[idx]
        results.append({
            "id": int(item_meta["id"]),
            "productDisplayName": str(item_meta["productDisplayName"]),
            "articleType": str(item_meta["articleType"]),
            "similarity_score": round(float(distances[0][i]), 4)
        })
        
    return {"results": results}
