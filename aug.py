import os
import pandas as pd
from PIL import Image, ImageEnhance
import numpy as np

# Пути
csv_path = 'C:/Users/User/Downloads/rtsd-public/full-gt.csv'
images_dir = 'C:/Users/User/Downloads/rtsd-public/full-frames/rtsd-frames'
labels_dir = 'labels'

os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df['sign_class'] = df['sign_class'].astype(str).str.strip()

class_counts = df['sign_class'].value_counts()
rare_classes_80 = set(class_counts[class_counts < 80].index)
rare_classes_50 = set(class_counts[class_counts < 50].index)
rare_classes_30 = set(class_counts[class_counts < 30].index)
print(f"Редкие классы <80: {len(rare_classes_80)},<50:{len(rare_classes_50)} <30: {len(rare_classes_30)}")

classes = ['2_1', '1_23', '1_17', '3_24_n40', '8_2_1', '5_20', '3_24_n20', '5_19_1', '5_16', '3_25_n20', '6_16', '7_15',
           '2_2', '2_4', '8_13_1', '4_2_1', '1_20_3', '1_25', '3_4_n8', '8_3_2', '3_4_1', '4_1_6', '4_2_3', '4_1_1',
           '1_33', '5_15_5', '3_27', '1_15', '4_1_2_1', '6_3_1', '8_1_1', '6_7', '5_15_3', '7_3', '1_19', '6_4',
           '8_1_4',
           '8_8', '1_16', '1_11_1', '6_6', '5_15_1', '7_2', '5_15_2', '7_12', '3_18', '5_6', '5_5', '7_4', '4_1_2',
           '8_2_2', '7_11',
           '3_24_n5', '1_22', '1_27', '2_3_2', '5_15_2_2', '1_8', '3_13_r5', '2_3', '8_3_3', '2_3_3', '7_7', '1_11',
           '8_13', '3_24_n30', '1_12_2', '1_20', '1_12', '3_24_n60', '3_24_n70', '3_24_n50', '3_32', '2_5', '3_1',
           '4_8_2', '3_20', '3_13_r4.5', '3_2', '2_3_6', '5_22', '5_18', '2_3_5', '7_5', '8_4_1', '3_13_r3.7',
           '3_14_r3.7', '1_2', '1_20_2', '4_1_4', '7_6', '8_1_3', '8_3_1', '4_3', '4_1_5', '8_2_3', '8_2_4', '3_24_n80',
           '1_31', '3_10', '4_2_2', '3_13_r2.5', '7_1', '3_28', '4_1_3', '5_4', '5_3', '3_25_n40', '3_13_r4', '6_8_2',
           '3_31', '6_2_n50', '3_24_n10', '3_25_n50', '1_21', '3_21', '1_13', '1_14', '6_2_n70', '2_3_4', '4_8_3',
           '6_15_2', '2_6', '3_18_2', '4_1_2_2', '1_7', '3_19', '1_18', '2_7', '8_5_4', '3_25_n80', '5_15_7', '5_14',
           '5_21', '1_1', '6_15_1', '3_4_n2', '8_6_4', '8_15', '4_5', '3_13_r4.2', '6_2_n60', '3_11_n23', '3_11_n9',
           '8_18', '8_4_4', '3_30', '5_7_1', '5_7_2', '1_5', '3_29', '6_15_3', '5_12', '3_16_n3', '3_13_r4.3', '1_30',
           '5_11', '1_6', '8_6_2', '6_8_3', '3_12_n10', '3_12_n6', '3_33', '3_11_n13', '3_14_r2.7', '3_16_n1', '8_4_3',
           '5_8', '3_11_n20', '3_11_n5', '8_14', '3_11_n8', '3_4_n5', '8_17', '3_6',
           '3_14_r3', '1_26', '3_12_n5', '8_5_2', '6_8_1', '5_17', '1_10', '3_13_r3.5', '3_13_r3.3', '3_13_r4.1',
           '3_11_n17', '8_16', '3_13_r3', '3_25_n70', '6_2_n20', '3_12_n3', '3_14_r3.5', '3_13_r3.9', '6_2_n40',
           '3_13_r5.2', '7_18', '7_14', '8_23',
           ]


def change_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def change_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def change_saturation(img, factor):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def add_gaussian_noise(img, mean=0, sigma=10):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, sigma, np_img.shape)
    np_img += noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def augment_image(img):
    # Случайные параметры аугментаций
    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.2)
    saturation_factor = np.random.uniform(0.8, 1.2)

    img = change_brightness(img, brightness_factor)
    img = change_contrast(img, contrast_factor)
    img = change_saturation(img, saturation_factor)

    # Вероятность добавить шум
    if np.random.rand() < 0.5:
        img = add_gaussian_noise(img, sigma=8)

    return img


def save_augmented_image_and_labels(orig_filename, img, records, classes, aug_idx):
    base_name = os.path.splitext(orig_filename)[0]
    aug_filename = f"aug_{aug_idx}_{base_name}.jpg"
    aug_image_path = os.path.join(images_dir, aug_filename)
    img.save(aug_image_path)

    img_w, img_h = img.size
    lines = []
    for _, row in records.iterrows():
        class_name = str(row['sign_class']).strip()
        if class_name not in classes:
            continue
        class_id = list(classes).index(class_name)

        # Масштабируем bbox координаты
        x_center = (row['x_from'] + row['width'] / 2) / img_w
        y_center = (row['y_from'] + row['height'] / 2) / img_h
        width = row['width'] / img_w
        height = row['height'] / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    aug_label_path = os.path.join(labels_dir, f"aug_{aug_idx}_{base_name}.txt")
    with open(aug_label_path, 'w') as f:
        f.write("\n".join(lines) + '\n')


augmented_count = 0

for filename in df['filename'].unique():
    image_path = os.path.join(images_dir, filename)
    if not os.path.exists(image_path):
        print(f"Пропущено (нет файла): {filename}")
        continue

    records = df[df['filename'] == filename]
    max_augs = 0
    for _, row in records.iterrows():
        cls = str(row['sign_class']).strip()
        count = class_counts.get(cls, 0)
        if count < 30:
            max_augs = max(max_augs, 4)
        elif count < 50:
            max_augs = max(max_augs, 2)
        elif count < 80:
            max_augs = max(max_augs, 1)

    if max_augs == 0:
        continue  # нет редких классов — пропускаем

    img = Image.open(image_path).convert('RGB')

    for i in range(max_augs):
        aug_img = augment_image(img)
        save_augmented_image_and_labels(filename, aug_img, records, classes, i)
        augmented_count += 1
        if augmented_count % 100 == 0:
            print(f"Аугментировано изображений: {augmented_count}")
print(f"Итого аугментировано изображений: {augmented_count}")
