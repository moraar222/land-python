# python code to run the main program

import sys 
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from pytesseract import Output
import pandas as pd
import re
import json
import requests

from utils import *

# Load the image
image_path = 'data/sample.jpg'
image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Extract the text from the image
text = pytesseract.image_to_string(preprocessed_image)

# Extract the text with bounding boxes
d = pytesseract.image_to_data(preprocessed_image, output_type=Output.DICT)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        preprocessed_image = cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Extract the text from the image
text = pytesseract.image_to_string(preprocessed_image)

# Extract the text with bounding boxes
d = pytesseract.image_to_data(preprocessed_image, output_type=Output.DICT)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        preprocessed_image = cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Extract the text from the image
text = pytesseract.image_to_string(preprocessed_image)