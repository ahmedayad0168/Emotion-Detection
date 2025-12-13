# 🤖 Real-Time Binary Emotion Detection (Happy/Sad)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for the binary classification of facial emotions (Happy and Sad). It features a robust real-time pipeline that combines the speed of traditional computer vision for face localization with the accuracy of deep learning for final classification.

-----

### 🌟 Project Goal: Deep Learning vs. Traditional CV

The core objective of this project was to compare and contrast two approaches to computer vision:

1.  **Deep Learning (CNN):** Training a custom model to learn complex emotional features directly from raw image data, providing high classification accuracy.
2.  **Traditional Computer Vision (OpenCV/Haar Cascades):** Utilizing pre-trained Haar Cascades for fast, efficient **face localization**—a critical preprocessing step—and contrasting its ability to perform the final classification task (e.g., simple smile detection vs. nuanced emotion classification).

The final solution is a **hybrid model**, leveraging the strengths of both: Haar Cascades for speed and the custom CNN for accuracy.

-----

### ✨ Key Features

  * **Custom CNN Architecture:** A sequential model trained from scratch for binary image classification.
  * **Real-Time Detection:** Utilizes the device camera to perform live emotion classification.
  * **Hybrid Pipeline:** Employs OpenCV's Haar Cascades for immediate face detection, then crops and feeds the localized face into the CNN for prediction.
  * **Performance:** Achieves high accuracy (e.g., 90%+) on the validation set.
  * **Robust Preprocessing:** Handles OpenCV's BGR-to-RGB color space conversion for seamless integration with TensorFlow/Keras.

-----

### 🧠 Model Architecture

The custom CNN is built with TensorFlow's Keras API. The architecture is designed to capture spatial hierarchies in the facial data:

  * **Input Layer:** Expects 256x256 RGB images.
  * **Convolutional Blocks:** Multiple `Conv2D` layers with `ReLU` activation and `MaxPooling2D` for feature extraction.
  * **Regularization:** Includes `Dropout` layers to prevent overfitting.
  * **Output Layer:** A final `Dense` layer with `Sigmoid` activation for binary classification (0 = Happy, 1 = Sad).

-----

### 📊 Results

The model was trained for **20 Epochs** and utilized an `EarlyStopping` callback to monitor validation loss, preventing overfitting.

| Metric | Training Result | Validation Result |
| :--- | :--- | :--- |
| **Final Accuracy** | 98.7% | 93.4% |
| **Final Loss** | 0.05 | 0.21 |

*(Insert a plot of the training history here, showing **Loss** and **Accuracy** over epochs)*

-----

### 🛠 Technology Stack

  * **Python** (3.8+)
  * **TensorFlow / Keras:** Deep Learning framework for model building and training.
  * **OpenCV (`cv2`):** Used for real-time video capture, face detection (Haar Cascades), and image preprocessing.
  * **NumPy:** Numerical operations.
  * **Jupyter Notebook:** Environment for development and visualization (`Emotion_Detection_CNN.ipynb`).

-----

### 🚀 Getting Started

#### Prerequisites

Ensure you have Python installed.

#### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/ahmedayad0168/Emotion-Detection.git
    cd Emotion-Detection
    ```

2.  **Install Dependencies:**

    ```bash
    pip install tensorflow opencv-python numpy jupyter matplotlib
    ```

#### Data Source

This project requires a dataset of Happy and Sad facial images, organized in subdirectories (`/data/happy`, `/data/sad`).

The model was trained on a dataset similar to the one available on Kaggle:

  * [Happy and Sad Image dataset for CNN - Kaggle](https://www.kaggle.com/datasets/saharnazyaghoobpoor/happy-and-sad-image)

-----

### 💡 Usage

The entire workflow, from data loading to real-time execution, is contained within the notebook.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Run the Notebook:** Open and execute all cells in `img_classifier.ipynb`.
3.  **Real-Time Demo:** The final cell will load the saved model (`new_model.h5`), initialize your camera, and display the live emotion detection feed.
      * Press `q` or `ESC` to exit the camera feed.



