import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
from emotions import detect_emotion_from_image, detect_emotion_from_camera


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotions Detector")
        self.root.geometry("1200x600")
        self.root.minsize(1000, 400)
        self.root.iconbitmap('icon.ico')

        self.root.configure(bg='#2C2C2C')
        self.button_style = {
            "height": 2,
            "width": 10,
            "bg": "#6E6E6E",
            "fg": "white",
            "font": ("Arial", 14),
            "relief": "solid",
            "bd": 2,
            "highlightbackground": "black",
            "highlightthickness": 2
        }

        self.running = False
        self.frame = None
        self.emotion = None
        self.lock = threading.Lock()

        self.main_menu()

    def main_menu(self):
        self.clear_window()

        title_label = tk.Label(self.root, text="Emotions Detector", font=("Arial", 40, "bold"), bg='#2C2C2C', fg="white")
        title_label.pack(pady=(20, 10))

        select_label = tk.Label(self.root, text="Select source", font=("Arial", 16), bg='#2C2C2C', fg="white")
        select_label.pack()

        button_frame = tk.Frame(self.root, bg='#2C2C2C')
        button_frame.pack(expand=True)

        self.photo_button = tk.Button(button_frame, text="Photo", command=self.open_photo, **self.button_style)
        self.photo_button.grid(row=0, column=0, padx=20, pady=10)

        self.camera_button = tk.Button(button_frame, text="Camera", command=self.open_camera, **self.button_style)
        self.camera_button.grid(row=0, column=1, padx=20, pady=10)

    def open_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            emotion, image, confidence = detect_emotion_from_image(file_path)
            self.display_result(image, emotion, confidence)

    def open_camera(self):
        from emotions import initialize_camera, release_camera, detect_emotion_from_camera

        self.clear_window()

        self.root.configure(bg='#2C2C2C')

        video_frame = tk.Frame(self.root, bg='#2C2C2C')
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        video_label = tk.Label(video_frame, width=800, height=600, bg='#2C2C2C')
        video_label.place(relx=0.0, rely=0.5, anchor="w")

        emotion_label_frame = tk.Frame(self.root, width=200, highlightbackground='black', highlightthickness=2)
        emotion_label_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        emotion_label = tk.Label(emotion_label_frame, text="", font=("Arial", 16), anchor="w", width=20, bg="#6E6E6E", fg="white")
        emotion_label.pack()


        try:
            initialize_camera()
        except RuntimeError as e:
            messagebox.showerror("Camera Error", str(e))
            return

        self.running = True

        def process_frame():
            while self.running:
                try:
                    frame, emotion = detect_emotion_from_camera()
                    with self.lock:
                        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.emotion = emotion
                except RuntimeError as e:
                    messagebox.showerror("Camera Error", str(e))
                    break

        def update_gui():
            if not self.running:
                return

            with self.lock:
                if self.frame is not None:
                    img = ImageTk.PhotoImage(image=Image.fromarray(self.frame))
                    video_label.imgtk = img
                    video_label.configure(image=img)
                    emotion_label.config(text=f"Emotion: {self.emotion}")

            if self.running:
                video_label.after(50, update_gui)

        threading.Thread(target=process_frame, daemon=True).start()
        update_gui()

        def on_close():
            self.running = False
            release_camera()
            self.main_menu()

        back_button = tk.Button(self.root, text="Back", command=on_close, **self.button_style)
        back_button.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)


    def display_result(self, image, emotion, confidence):
        self.clear_window()

        img = Image.fromarray(image)

        self.root.configure(bg='#2C2C2C')

        max_width, max_height = 800, 600
        img.thumbnail((max_width, max_height))

        imgtk = ImageTk.PhotoImage(image=img)

        img_label = tk.Label(self.root, image=imgtk)
        img_label.imgtk = imgtk
        img_label.pack(side=tk.LEFT, padx=20, pady=20)

        confidence_percentage = f"Confidence: {confidence * 100:.2f}%"

        emotion_label = tk.Label(self.root, text=f"Emotion: {emotion}\n{confidence_percentage}", font=("Arial", 16), bg="#6E6E6E", fg="white", highlightbackground='black', highlightthickness=2)
        emotion_label.pack(side=tk.RIGHT, padx=20)

        back_button = tk.Button(self.root, text="Back", command=self.main_menu, **self.button_style)
        back_button.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()