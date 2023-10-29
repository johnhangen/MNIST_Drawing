import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess_image(image_path: str) -> np.array:
    '''
    Preprocesses the image to be fed into the model.
    
    Parameters:
        image_path (str): Path to the image.
        
    Returns:
        flat_img_array (np.array): 1D array of the image.
    '''
    try:
        img = Image.open(image_path)
    except:
        print('Image not found. Please check the path.')
        return None

    img = img.resize((28, 28))
    img = img.convert('L')

    img_array = (np.array(img) / 255.0) - 0.5
    flat_img_array = img_array.reshape((1, 28 * 28))
    
    return flat_img_array

def visualize_vector(image_vector, prediction:int = 999):
    '''
    Visualizes the image vector.
    
    Parameters:
        image_vector (np.array): 1D array of the image.
        
    Returns:
        None
    '''
    assert image_vector.shape == (1,784)

    image_2d = image_vector.reshape((28, 28))

    plt.imshow(image_2d, cmap='gray')
    if prediction != 999:
        plt.title(f'Prediction: {prediction}')
    #plt.axis('off')
    plt.show()

def load_model_h5(model_path: str):
    '''
    Loads the model from the given path.
    
    Parameters:
        model_path (str): Path to the model.
    
    Returns:
        model (tensorflow.keras.models.Model): Model object.
    '''
    try:
        print('Loading model...')
        model = tf.keras.models.load_model(model_path)
        print('Model loaded.')
        return model
    except:
        print('Model not found. Please check the path.')

class SimpleDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Drawing App")

        self.canvas = tk.Canvas(root, bg="white", width=600, height=400)
        self.canvas.pack(pady=20)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = 'black'
        self.drawings = Image.new('RGB', (600, 400), 'white')
        self.drawer = ImageDraw.Draw(self.drawings)

        menu = tk.Menu(root)
        root.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Clear", command=self.clear_canvas)
        file_menu.add_command(label="Save", command=self.save_canvas)
        file_menu.add_command(label="Exit", command=root.destroy)

    def draw(self, event):
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line((self.old_x, self.old_y, x, y), width=self.line_width, fill=self.color, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.drawer.line((self.old_x, self.old_y, x, y), fill=self.color, width=self.line_width)
        self.old_x = x
        self.old_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawings = Image.new('RGB', (600, 400), 'white')
        self.drawer = ImageDraw.Draw(self.drawings)

    def save_canvas(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            self.drawings.save(file_path)
        
        self.clear_canvas()
        self.root.destroy()

    def reset_old_coordinates(self, event):
        self.old_x = None
        self.old_y = None


def main():
    root = tk.Tk()
    app = SimpleDrawingApp(root)
    root.mainloop()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    image_path = r'images\user_test.png'
    model_path = r'models\model_savedmodel'
    processed_image = preprocess_image(image_path)
    model = load_model_h5(model_path)

    prediction = model.predict(processed_image)

    print(f'Prediction: {np.argmax(prediction)}')
    visualize_vector(processed_image, np.argmax(prediction))

if __name__ == "__main__":
    main()
