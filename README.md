\# **Smart Waste Classification**



\## Project Description

This project uses a Convolutional Neural Network (CNN) to classify 12 types of waste for sustainability.  

It automates waste segregation using AI, which can help reduce environmental pollution and support recycling initiatives.



\## Dataset

\- Kaggle Garbage Classification Dataset: \[Link](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)  

\- Classes: metal, white-glass, brown-glass, paper, trash, cardboard, clothes, biological, shoes, plastic, battery, green-glass



\## Project Structure

Smart-Waste-Classification/

├─ notebook.ipynb # Colab notebook with training and evaluation

├─ waste\_classification\_model.h5 # Trained CNN model

├─ README.md # Project explanation

└─ sample\_images/ # Optional test images



\## How to Test New Images

```python

from tensorflow.keras.models import load\_model

from tensorflow.keras.preprocessing import image

import numpy as np



\# Load the trained model

model = load\_model('waste\_classification\_model.h5')



\# Load and preprocess a sample image

img = image.load\_img('sample\_images/plastic.jpg', target\_size=(128,128))

img\_array = image.img\_to\_array(img)/255.0

img\_array = np.expand\_dims(img\_array, axis=0)



\# Make prediction

prediction = np.argmax(model.predict(img\_array))

classes = \['metal', 'white-glass', 'brown-glass', 'paper', 'trash', 

&nbsp;          'cardboard', 'clothes', 'biological', 'shoes', 'plastic', 'battery', 'green-glass']



\# Print predicted class

print("Predicted Waste Type:", classes\[prediction])



**Results**



Training Accuracy: ~85–90%

Validation Accuracy: ~80–85%

Confusion matrix and classification report are included in the notebook for detailed evaluation.

Model is able to correctly classify images into 12 categories of waste with high precision and recall.



**Future Scope**



Build a web app (Streamlit or Flask) for live image predictions.

Increase dataset size and epochs for better accuracy.

Deploy on mobile devices for real-time waste classification in communities.

Implement automatic alerts or sorting systems in smart bins using this model.

