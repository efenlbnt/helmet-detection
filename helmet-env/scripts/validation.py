import os
import shutil
import random

def create_val_split(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, val_split=0.2):
    # Eğitim ve doğrulama veri setleri için klasörleri oluştur
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Tüm eğitim görüntü dosyalarını al
    image_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]
    
    # Doğrulama için ayrılacak dosya sayısı
    num_val_images = int(len(image_files) * val_split)
    
    # Doğrulama için ayrılacak dosyaları rastgele seç
    val_images = random.sample(image_files, num_val_images)

    for image in val_images:
        # Görüntü dosyasını taşı
        shutil.move(os.path.join(train_images_dir, image), os.path.join(val_images_dir, image))
        
        # Etiket dosyasını taşı
        label = image.replace('.jpg', '.txt')
        shutil.move(os.path.join(train_labels_dir, label), os.path.join(val_labels_dir, label))

# Klasör yolları
train_images_dir = 'dataset\labels\train'
train_labels_dir = 'dataset\labels\train'
val_images_dir = 'dataset\labels\val'
val_labels_dir = 'dataset\labels\val'

# Doğrulama veri setini oluştur
create_val_split(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir)
