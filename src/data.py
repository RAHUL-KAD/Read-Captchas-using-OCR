import os
from pathlib import Path

# Path to data directory
data = Path('/home/rahul/Desktop/ml_project/Deep learning/Read-Captchas-using-OCR/dataset')

# Get the list of all the images
images = sorted(list(map(str, list(data.glob("*.png")))))

# images labels are image names(except file extensions)
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

# Each label contain characters
characters = set(char for label in labels for char in label)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Maximun length of any captche in the dataset
max_length = max([len(label) for label in labels])
