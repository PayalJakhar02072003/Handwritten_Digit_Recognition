 ##✍️ Handwritten Digit Recognition  ##

This project is a Handwritten Digit Recognition System built using Convolutional Neural Networks (CNNs) on the MNIST dataset. The model can classify grayscale images of handwritten digits (0–9) with high accuracy and also supports testing with custom images.

🚀 Features

Recognizes handwritten digits (0–9).

Trained on the popular MNIST dataset (70,000 grayscale images).

Built using Convolutional Neural Networks (CNNs).

Allows testing with your own custom digit images.

Easy-to-run Jupyter Notebook implementation.

📂 Dataset

Dataset Name: MNIST Handwritten Digits

Description: Contains 60,000 training images and 10,000 testing images of handwritten digits (0–9).

Image Size: 28x28 pixels (grayscale).

Automatically available via tensorflow.keras.datasets.

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib (for visualization)

Jupyter Notebook

⚙️ Installation

Clone this repository:

git clone https://github.com/PayalJakhar02072003/Handwritten_Digit_recognition.git
cd Handwritten_Digit_recognition


Install the required libraries:

pip install tensorflow numpy matplotlib


Open the notebook:

jupyter notebook Handwritten_Digit_recognition.ipynb

▶️ Usage
1. Train the Model

Run all cells in the notebook. The model will:

Load and preprocess the MNIST dataset.

Train a CNN model for 10–20 epochs.

Display training accuracy and loss.

# Training example
model.fit(X_train, y_train, epochs=10)

2. Test with Custom Images

You can test with your own handwritten digit image:

from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

image_path = '/content/image.png'   # Path to your image
image = load_img(image_path, target_size=(28,28), color_mode='grayscale')
image_array = img_to_array(image) / 255.0
image_array = image_array.reshape((1,28,28,1))

prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)

📊 Output / Results

The CNN achieves high accuracy on MNIST test data (typically > 98%).

Example output:

Predicted class: 7


Custom images can also be tested by providing paths.



📬 Contact

👩 Author: Payal Jakhar
🔗 GitHub: PayalJakhar02072003
