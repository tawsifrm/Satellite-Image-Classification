import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("Model.h5")

# Define the class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Load an image from the test set
img = load_img("kaggle/input/satellite-image-classification/data/cloudy/train_3995.jpg", target_size=(255, 255))

# Convert the image to an array
img_array = img_to_array(img)
img_array.shape
img_array = img_array / 255.0
img_array = np.reshape(img_array, (1, 255, 255, 3))

# Get the model predictions
predictions = model.predict(img_array)

# Get the class index with the highest predicted probability
class_index = np.argmax(predictions[0])

# Get the predicted class label
predicted_label = class_names[class_index]

# Display the resized image with predicted text below
fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')  # Hide axis

# Add predicted text below the image
ax.text(0.5, -0.1, "Predicted: {}".format(predicted_label), transform=ax.transAxes,
        horizontalalignment='center', verticalalignment='center', fontsize=12)

plt.show()