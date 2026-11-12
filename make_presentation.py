import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

os.makedirs('presentation', exist_ok=True)
os.makedirs('val_data/images', exist_ok=True)

df = pd.read_csv('catalog_meta.csv')
df['baseColour'] = df['baseColour'].fillna('Unknown')
df['target_class'] = df['articleType'].astype(str) + '_' + df['baseColour'].astype(str)
index = faiss.read_index('fashion_catalog.index')

np.random.seed(42)
sample_indices = np.random.choice(len(df), size=1500, replace=False)
sample_embeddings = np.array([index.reconstruct(int(i)) for i in sample_indices], dtype=np.float32)

distances, neighbors = index.search(sample_embeddings, 11)

intra_sims = []
inter_sims = []
k_values = [1, 2, 3, 5, 10]
recalls_category = {k: 0 for k in k_values}
recalls_exact = {k: 0 for k in k_values}

for i, q_idx in enumerate(sample_indices):
    q_cat = df.iloc[q_idx]['articleType']
    q_target = df.iloc[q_idx]['target_class']
    
    top_indices = neighbors[i][1:11]
    top_dists = distances[i][1:11]
    
    top_cats = [df.iloc[idx]['articleType'] for idx in top_indices]
    top_targets = [df.iloc[idx]['target_class'] for idx in top_indices]
    
    for k in k_values:
        if q_cat in top_cats[:k]:
            recalls_category[k] += 1
        if q_target in top_targets[:k]:
            recalls_exact[k] += 1
            
    for dist, target in zip(top_dists, top_targets):
        if target == q_target:
            intra_sims.append(dist)
        else:
            inter_sims.append(dist)

total = len(sample_indices)

plt.figure(figsize=(9, 5), dpi=300)
plt.hist(inter_sims[:3000], bins=50, alpha=0.6, color='#E53935', label='Inter-class (Разные категории)', density=True)
plt.hist(intra_sims[:3000], bins=50, alpha=0.7, color='#1E88E5', label='Intra-class (Одинаковые: Тип + Цвет)', density=True)
plt.axvline(x=0.55, color='black', linestyle='--', linewidth=1.5, label='Порог разделения (Margin ≈ 0.55)')
plt.title('Распределение косинусного сходства (Cosine Similarity Distribution)', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Косинусное сходство (Cosine Similarity)', fontsize=10)
plt.ylabel('Плотность распределения (Density)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('presentation/3_similarity_distribution.png')
plt.close()
print("Saved 3_similarity_distribution.png")

plt.figure(figsize=(8, 5), dpi=300)
rec_cat_pct = [recalls_category[k]/total*100 for k in k_values]
rec_ex_pct = [recalls_exact[k]/total*100 for k in k_values]

plt.plot(k_values, rec_cat_pct, marker='o', linewidth=2.2, color='#1E88E5', label='Recall@K: Категория (articleType)')
plt.plot(k_values, rec_ex_pct, marker='s', linewidth=2.2, color='#43A047', label='Recall@K: Точное совпадение (Тип + Цвет)')

for k, val in zip(k_values, rec_cat_pct):
    plt.annotate(f'{val:.1f}%', (k, val), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8, fontweight='bold', color='#1E88E5')

for k, val in zip(k_values, rec_ex_pct):
    plt.annotate(f'{val:.1f}%', (k, val), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, fontweight='bold', color='#43A047')

plt.title('Кривая качества поиска Recall@K (CMC Curve)', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Число возвращаемых результатов (K)', fontsize=10)
plt.ylabel('Точность ранжирования Recall (%)', fontsize=10)
plt.ylim(min(rec_ex_pct) - 10, 105)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=True, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('presentation/4_recall_at_k.png')
plt.close()
print("Saved 4_recall_at_k.png")

print("Exporting local validation mini-benchmark (val_data/)...")
dataset = load_dataset('ashraq/fashion-product-images-small', split='train')

top_classes = [
    'Tshirts_Black', 'Tshirts_White', 'Tshirts_Blue',
    'Watches_Black', 'Sports Shoes_White', 'Casual Shoes_Black',
    'Shirts_Blue', 'Jeans_Blue', 'Casual Shoes_Brown', 'Sunglasses_Black'
]

val_rows = []
for cls in top_classes:
    cls_indices = df[df['target_class'] == cls].index[:30].tolist()
    for idx in cls_indices:
        item = df.iloc[idx]
        img = dataset[idx]['image']
        filename = f"{item['id']}.jpg"
        img.save(os.path.join('val_data/images', filename))
        val_rows.append({
            'id': int(item['id']),
            'articleType': str(item['articleType']),
            'baseColour': str(item['baseColour']),
            'target_class': str(item['target_class']),
            'productDisplayName': str(item['productDisplayName']),
            'image_filename': filename
        })

val_df = pd.DataFrame(val_rows)
val_df.to_csv('val_data/labels.csv', index=False)
print(f"Exported {len(val_df)} local images to val_data/images/ and val_data/labels.csv")
