import cv2
import os
import pandas as pd

# Görüntüleri ve etiketleri yüklemek için yollar
image_folder = 'helmet-env/data/images'
output_folder = 'helmet-env/data/boxed_images'
annotations_df = pd.read_csv('helmet-env/data/annotations.csv')

# Çıktı klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Görüntüler üzerine kutu çizme ve kaydetme
for filename in annotations_df['filename'].unique():
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    if image is not None:
        sub_df = annotations_df[annotations_df['filename'] == filename]
        for _, row in sub_df.iterrows():
            # Sınırlayıcı kutuyu çiz
            cv2.rectangle(image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
        # Tüm kutular çizildikten sonra görüntüyü kaydet
        cv2.imwrite(os.path.join(output_folder, filename), image)