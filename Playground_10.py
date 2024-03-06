import datetime
import csv
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("800x600")
        self.root.configure(background="black")

        self.header_label = tk.Label(root, text="Face Recognition App", font=("Helvetica", 24), fg="white", bg="black")
        self.header_label.pack(pady=20)

        self.create_input_widgets()
        self.create_canvas()

        self.detect_faces()

    def create_input_widgets(self):
        self.name_label = tk.Label(self.root, text="Enter your name:", font=("Helvetica", 14), fg="white", bg="black")
        self.name_label.pack()

        self.name_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.name_entry.pack()

        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 14), fg="white", bg="black")
        self.status_label.pack()

    def create_canvas(self):
        self.canvas_frame = tk.Frame(self.root, bg="white", bd=5)
        self.canvas_frame.pack(pady=20)

        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=480)
        self.canvas.pack()

    def save_data(self, name):
        if name:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%p")
            image_filename = f"{name}_{timestamp}.jpg"
            cv2.imwrite(image_filename, cv2.cvtColor(self.last_frame_rgb, cv2.COLOR_RGB2BGR))  # Save image
            self.save_to_csv(name, timestamp, image_filename)
            messagebox.showinfo("Success", "Name, timestamp, and image saved successfully!")
        else:
            messagebox.showwarning("Error", "Please enter your name before saving the image.")

    def save_to_csv(self, name, timestamp, image_filename):
        formatted_timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d_%I-%M-%p").strftime("%Y-%m-%d %I:%M %p")
        with open("data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, formatted_timestamp, image_filename])

    def detect_faces(self):
        # Load the Haar cascade classifier for detecting frontal faces and smiles
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Initialize the video capture object
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or an index for another camera

        while True:
            self.face_detected = False  # Reset face detection flag at the beginning of each loop iteration
            self.smile_detected = False  # Reset smile detection flag at the beginning of each loop iteration

            # Capture a frame from the webcam
            ret, frame = cap.read()

            if ret:
                # Convert the frame to RGB format
                self.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw a green rectangle around each detected face
                self.face_detected = len(faces) > 0
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=25, minSize=(30, 30))
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                        self.smile_detected = True

                # Convert the frame to RGB format and display it on the canvas
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

                # Update the status label based on whether faces are detected
                if self.face_detected:
                    if self.smile_detected:
                        self.status_label.config(text="Face Detected - Smile Detected", fg="green")
                        name = self.name_entry.get()
                        self.save_data(name)
                    else:
                        self.status_label.config(text="Face Detected - Smile Not Detected", fg="red")
                else:
                    self.status_label.config(text="No Face Detected", fg="red")

                # Update the window and handle events
                self.root.update_idletasks()
                self.root.update()

            # Exit the loop if the window is closed or 'q' key is pressed
            if cv2.waitKey(1) == ord('q') or not self.root.winfo_exists():
                break

        # Release the capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
