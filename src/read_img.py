from PIL import Image
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
    

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    image_path = r'images\user_test.png'
    model_path = r'models\model_savedmodel'
    processed_image = preprocess_image(image_path)
    model = load_model_h5(model_path)

    prediction = model.predict(processed_image)

    print(f'Prediction: {np.argmax(prediction)}')
    visualize_vector(processed_image, np.argmax(prediction))
    

if __name__ == '__main__':
    main()
