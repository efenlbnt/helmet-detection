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
root.geometry("1200x800")
root.configure(bg='#f0f0f0')

# Küresel değişkenler
global_image = None

def load_image():
    global global_image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        global_image = image.copy()
        
        # Orijinal resmi sıkıştırarak göster
        resized_image = cv2.resize(image, (400, 400))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        label_original_image.config(image=image_tk)
        label_original_image.image = image_tk
        button_detect.config(state=tk.NORMAL)

def detect_helmet():
    global global_image
    if global_image is not None:
        image = global_image.copy()
        results = model(image)
        
        detections = results.xyxy[0].cpu().numpy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Helmet: {conf:.2f}" if int(cls) == 0 else f"No Helmet: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Tespit edilen sonuçları aynı boyutta göster
        resized_image = cv2.resize(image, (400, 400))
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label_detected_image.config(image=image_tk)
        label_detected_image.image = image_tk
        button_reset.config(state=tk.NORMAL)

def reset_application():
    label_original_image.config(image='')
    label_detected_image.config(image='')
    button_detect.config(state=tk.DISABLED)
    button_reset.config(state=tk.DISABLED)
    global global_image
    global_image = None

# Arayüz elemanları
frame = tk.Frame(root, bg='#f0f0f0')
frame.pack(pady=20)

label_original_image = tk.Label(frame, bg='#f0f0f0')
label_original_image.pack(side=tk.LEFT, padx=10)

label_detected_image = tk.Label(frame, bg='#f0f0f0')
label_detected_image.pack(side=tk.RIGHT, padx=10)

button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(pady=20)

button_load = tk.Button(button_frame, text="Fotoğraf Yükle", command=load_image, bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'))
button_load.pack(side=tk.LEFT, padx=10)

button_detect = tk.Button(button_frame, text="Tespit Et", command=detect_helmet, state=tk.DISABLED, bg='#f44336', fg='white', font=('Arial', 12, 'bold'))
button_detect.pack(side=tk.LEFT, padx=10)

button_reset = tk.Button(button_frame, text="Yeniden Başla", command=reset_application, state=tk.DISABLED, bg='#2196F3', fg='white', font=('Arial', 12, 'bold'))
button_reset.pack(side=tk.LEFT, padx=10)

root.mainloop()
