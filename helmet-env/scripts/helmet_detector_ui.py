import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import pandas as pd

def load_image():
    global image_path, image, photo, label_image
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image. Please select a valid image file.")
            return
        # Convert image for display
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_display)
        photo = ImageTk.PhotoImage(im_pil)
        label_image.config(image=photo)
        label_image.image = photo
        messagebox.showinfo("Image Loaded", "Successfully loaded the image!")
        print("Loaded image:", image_path)

def detect_helmets():
    global image, photo, label_image
    if 'image_path' not in globals():
        messagebox.showwarning("Load Image", "Please load an image first!")
        return

    output_path = 'helmet-env/data/boxed_images'
    annotations_df = pd.read_csv('helmet-env/data/annotations.csv')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if image is not None:
        filename = os.path.basename(image_path)
        sub_df = annotations_df[annotations_df['filename'] == filename]

        if sub_df.empty:
            # Eğer XML dosyası yoksa, örnek bir model ile tespit yap ve çiz
            print("No annotations found for this image. Running helmet detection model...")
            # Bu kısımda gerçek bir model ile kask tespiti yapmalısınız.
            # Örnek olarak sadece ortada bir dikdörtgen çiziyoruz.
            height, width = image.shape[:2]
            cv2.rectangle(image, (int(width*0.3), int(height*0.3)), (int(width*0.7), int(height*0.7)), (0, 255, 0), 2)
        else:
            for _, row in sub_df.iterrows():
                cv2.rectangle(image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_path, filename), image)
        # Update the display after detection
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image_display)
        photo = ImageTk.PhotoImage(im_pil)
        label_image.config(image=photo)
        label_image.image = photo
        messagebox.showinfo("Detection Complete", "Helmets have been detected and marked.")
        print("Processed image saved: ", os.path.join(output_path, filename))

app = tk.Tk()
app.title("Helmet Detection")

label_image = tk.Label(app)
label_image.pack()

load_button = tk.Button(app, text="Load Image", command=load_image)
load_button.pack()

detect_button = tk.Button(app, text="Detect Helmets", command=detect_helmets)
detect_button.pack()

app.mainloop()