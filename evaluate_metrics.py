import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(EmbeddingNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Вычисления на: {device}")

model = EmbeddingNet(embedding_dim=512)
model.load_state_dict(torch.load('mobilenet_v3_embedding.pth', map_location='cpu'))
model.to(device)
model.eval()

print("Загрузка датасета...")
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
dataset = dataset.shuffle(seed=42).select(range(2000))
selected_data = dataset.to_pandas() 

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

embeddings = []
print("Извлечение векторов...")
with torch.no_grad():
    for item in tqdm(dataset):
        img_pil = item['image'].convert("RGB")
        img_np = np.array(img_pil)
        tensor = val_transforms(image=img_np)['image'].unsqueeze(0).to(device)
        
        emb = model(tensor).cpu().numpy()
        embeddings.append(emb[0])

embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

faiss.omp_set_num_threads(1)
index = faiss.IndexFlatIP(512)
index.add(embeddings)

top1_correct = 0
top5_correct = 0

print("\nПодсчет метрик...")
distances, indices = index.search(embeddings, 6)

for i in range(len(selected_data)):
    query_category = selected_data.iloc[i]['articleType']
    neighbor_indices = indices[i][1:6]
    neighbor_categories = [selected_data.iloc[idx]['articleType'] for idx in neighbor_indices]
    
    if neighbor_categories[0] == query_category:
        top1_correct += 1
        
    if query_category in neighbor_categories:
        top5_correct += 1

print("-" * 30)
print(f"Top-1 Accuracy: {top1_correct / len(selected_data) * 100:.2f}%")
print(f"Top-5 Precision: {top5_correct / len(selected_data) * 100:.2f}%")
print("-" * 30)
