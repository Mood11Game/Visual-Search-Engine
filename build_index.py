import onnxruntime as ort
import numpy as np
import faiss
import pandas as pd
from datasets import load_dataset

# 1. Загружаем ONNX-модель через CPU-провайдер (для M2 работает отлично)
session = ort.InferenceSession("visual_search.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 2. Подтягиваем датасет локально
print("Загружаем датасет каталога...")
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
selected_data = dataset.to_pandas()

# Сохраняем метаданные, чтобы при поиске понимать, что за товар мы нашли
selected_data.to_csv("catalog_meta.csv", index=False)

# 3. Настраиваем FAISS
embedding_dim = 512
# Используем Inner Product (faiss.IndexFlatIP), так как наша сеть делает L2-нормализацию векторов.
# Векторы на единичной сфере + скалярное произведение = косинусное сходство.
index = faiss.IndexFlatIP(embedding_dim)

# 4. Препроцессинг картинки на чистом NumPy (как при обучении)
def preprocess(image_pil):
    img = image_pil.convert("RGB").resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Нормализация ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # Меняем порядок осей: HWC -> CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

# 5. Извлекаем векторы батчами (чтобы не забить оперативку)
batch_size = 128
embeddings_list = []

print("Извлекаем векторы (это займет пару минут)...")
for i in range(0, len(dataset), batch_size):
    batch_images = dataset[i : i + batch_size]["image"]
    
    # Подготавливаем батч тензоров
    batch_tensors = np.stack([preprocess(img) for img in batch_images])
    
    # Делаем инференс через ONNX
    batch_embeddings = session.run(None, {input_name: batch_tensors})[0]
    embeddings_list.append(batch_embeddings)
    
    if (i + batch_size) % 1280 == 0:
        print(f"Обработано {i + batch_size} / {len(dataset)} картинок")

# 6. Добавляем в векторную базу и сохраняем на диск
all_embeddings = np.vstack(embeddings_list)
index.add(all_embeddings)
faiss.write_index(index, "fashion_catalog.index")

print(f"Готово! Индекс сохранен. Всего товаров в базе: {index.ntotal}")

