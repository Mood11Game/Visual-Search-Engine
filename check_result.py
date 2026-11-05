import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# 1. Подтягиваем локальный кэш датасета и метаданные
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
df_meta = pd.read_csv("catalog_meta.csv")

# 2. Вставь сюда ID из ответа Swagger (из поля "id")
# Например: [47359, 11674, 31120]
target_ids = [37586, 34096, 27444, 27445, 34683] # Замени на свои цифры!

# 3. Отрисовываем картинки
plt.figure(figsize=(15, 5))
for i, item_id in enumerate(target_ids):
    # Находим порядковый индекс строки, где id совпадает с искомым
    idx = df_meta.index[df_meta['id'] == item_id].tolist()[0]
    
    # Достаем саму картинку и ее название
    img = dataset[idx]['image']
    title = df_meta.loc[idx, 'productDisplayName']
    
    plt.subplot(1, len(target_ids), i + 1)
    plt.imshow(img)
    plt.title(f"ID: {item_id}\n{title[:25]}...", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()
