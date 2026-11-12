import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
df_meta = pd.read_csv("catalog_meta.csv")

target_ids = [7566, 27969, 9989, 50195, 50194]

plt.figure(figsize=(15, 5))
for i, item_id in enumerate(target_ids):
    idx = df_meta.index[df_meta['id'] == item_id].tolist()[0]
    img = dataset[idx]['image']
    title = df_meta.loc[idx, 'productDisplayName']
    
    plt.subplot(1, len(target_ids), i + 1)
    plt.imshow(img)
    plt.title(f"ID: {item_id}\n{title[:25]}...", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()
