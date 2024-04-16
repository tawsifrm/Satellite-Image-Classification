# Satellite Image Classification

This repository contains code for building a deep learning model to classify satellite images into different categories such as Cloudy, Desert, Green Area, and Water.

## What It Does

The model is designed to take satellite images as input and predict the category of the image, providing insights into the type of terrain or environment captured in the satellite data.

## How It Works

1. **Data Preparation:**
   - The data can be obtained from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data), where it's organized into folders representing different categories (e.g., Cloudy, Desert, Green Area, Water).
   - Image paths and labels are extracted from the folder structure and saved into a CSV file for easier handling.

2. **Data Preprocessing:**
   - Images are resized and preprocessed using techniques like rescaling, shear, zoom, flip, and rotation to enhance the model's ability to generalize.

3. **Model Architecture:**
   - A deep learning model is built using Convolutional Neural Networks (CNNs) for feature extraction.
   - The model architecture includes multiple Conv2D layers with activation functions like ReLU, MaxPooling layers for spatial downsampling, a Flatten layer to convert the 2D feature maps into a 1D vector, and Dense layers for classification.
   - Dropout is used to prevent overfitting by randomly disabling neurons during training.

4. **Training and Evaluation:**
   - The model is trained using the prepared dataset with a specified number of epochs.
   - Training and validation metrics such as loss and accuracy are monitored and visualized using matplotlib.

5. **Model Deployment:**
   - After training, the model is saved as a `.h5` file for future use.

6. **Prediction:**
   - The trained model is used to make predictions on new satellite images.
   - Users can change the image path in `run_model.py` to any satellite image of their choice, as long as it is the right size (255x255 pixels).

## Tools and Methods Used

- **Python Libraries:** pandas, os, numpy, sklearn, keras, PIL, matplotlib, tkinter
- **Data Augmentation:** ImageDataGenerator for generating augmented images during training.
- **Model Building:** Sequential model with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
- **Model Evaluation:** Metrics such as accuracy and loss are used to evaluate model performance.
- **Visualization:** Matplotlib is used to visualize training and validation metrics.

## Usage

1. Clone the repository to your local machine.
2. Install the required libraries.
3. Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data) and organize it into folders as per the provided structure.
4. Run `create_model.py` to train the model and save it as `Model.h5`.
5. Use `run_model.py` to load the trained model and make predictions on new satellite images by changing the image path to your desired satellite image of the right size (255x255 pixels).

## Example Result
![predictionui](https://github.com/tawsifrm/Satellite-Image-Classification/assets/121325051/9b81522d-7aa5-46af-87d9-22aff9a2e32d)
