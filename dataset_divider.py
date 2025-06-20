import os
import shutil
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

# Папки
source_images = r'C:\Users\User\Downloads\rtsd-public\full-frames\rtsd-frames'
source_labels = r'C:\Users\User\Desktop\sk-learn\labels'
output_dir = 'dataset'
train_ratio = 0.8
empty_label_count = 0
total_labels = 0
print("sdfsdfsd")

# Сначала собираем статистику: сколько изображений на каждый класс
class_to_images = defaultdict(list)

for fname in os.listdir(source_labels):
    if not fname.endswith('.txt'):
        continue
    label_path = os.path.join(source_labels, fname)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        continue
    # Считаем по первому классу (можно улучшить до мультиклассовой схемы)
    first_class = int(lines[0].split()[0])
    base_name = os.path.splitext(fname)[0]
    class_to_images[first_class].append(base_name)

# Удаляем классы с < 2 изображениями
image_class_list = []
for cls, images in class_to_images.items():
    if len(images) >= 2:
        for img in images:
            image_class_list.append((img, cls))
    else:
        print(f"Класс {cls} встречается только в {len(images)} изображении — удалён из датасета")

# Готовим X и y
X = [name for name, cls in image_class_list]
y = [cls for name, cls in image_class_list]

# Стратифицированное разделение
splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=42)
train_idx, val_idx = next(splitter.split(X, y))

train_files = [X[i] for i in train_idx]
val_files = [X[i] for i in val_idx]

# Создание папок
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# Копирование
def copy_split(files, split):
    for base_name in files:
        # Копируем изображение
        for ext in ['.jpg', '.png', '.jpeg']:
            img_src = os.path.join(source_images, base_name + ext)
            if os.path.exists(img_src):
                img_dst = os.path.join(output_dir, 'images', split, base_name + ext)
                shutil.copyfile(img_src, img_dst)
                break
        else:
            print(f"Изображение не найдено: {base_name}")
            continue

        # Копируем аннотацию
        label_src = os.path.join(source_labels, base_name + '.txt')
        label_dst = os.path.join(output_dir, 'labels', split, base_name + '.txt')
        if os.path.exists(label_src):
            shutil.copyfile(label_src, label_dst)
        else:
            open(label_dst, 'w').close()

# Копируем файлы
copy_split(train_files, 'train')
copy_split(val_files, 'val')

print("Датасет успешно разделён с учётом классов, встречающихся хотя бы дважды.")
