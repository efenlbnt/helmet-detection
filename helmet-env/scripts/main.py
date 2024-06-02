import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import os

# Model dosyasının doğru yerde olup olmadığını kontrol edin
model_path = r'C:\Users\Efe\Documents\GitHub\helmet-detection\yolov5\runs\train\exp43\weights\best.pt'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

# Cihazı kontrol edin (GPU veya CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLOv5 modelini yükleyin ve cihazda çalıştırın
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(device)

# Tkinter uygulaması
root = tk.Tk()
root.title("Kask Tespiti")
root.geometry("800x600")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        label_image.config(image=image_tk)
        label_image.image = image_tk

        # Fotoğrafı YOLOv5 modeli ile analiz edin
        results = model(file_path)
        
        # Tespit edilen sonuçları alın
        detections = results.xyxy[0].cpu().numpy()  # Sonuçları CPU'ya taşıyın ve numpy array olarak alın
        
        # Fotoğraf üzerine kutular ve etiketler ekleyin
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Helmet: {conf:.2f}" if int(cls) == 0 else f"No Helmet: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Güncellenmiş fotoğrafı gösterin
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label_image.config(image=image_tk)
        label_image.image = image_tk

label_image = tk.Label(root)
label_image.pack()

button_load = tk.Button(root, text="Fotoğraf Yükle", command=load_image)
button_load.pack()

root.mainloop()
