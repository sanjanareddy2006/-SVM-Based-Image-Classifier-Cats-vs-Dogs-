import os
import cv2
import numpy as np
from tqdm import tqdm

def load_data(data_dir, max_images=500):
    images = []
    labels = []
    categories = ['Cat', 'Dog']

    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        count = 0
        for file in tqdm(os.listdir(folder), desc=f"🔄 Loading {category}s"):
            if count >= max_images:
                break
            file_path = os.path.join(folder, file)
            try:
                img = cv2.imread(file_path)
                if img is None or img.size == 0:
                    raise Exception("Image is empty or unreadable")
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(label)
                count += 1
            except Exception as e:
                print(f"{category}/{file} skipped: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
