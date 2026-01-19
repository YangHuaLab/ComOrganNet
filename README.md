Project Description:
- Python tool for extracting image features, supporting batch processing.

Feature List
- Function 1: Image input format: tiff
- Function 2: Output the feature vector as a '.csv' file, and output the classified TSNE image
- Function 3: CPU/GPU compatible to run

Install the necessary packages
pip install -r requirements.txt

Preprocessing of image data before entering the model
1. Use the classify.ipynb code in the preprocess folder to mix the ch1-ch4 image folders and classify them under the four folders of ch1, ch2, ch3, and ch4
2. Use the crop.ipynb code in the preprocess folder to crop the images of the 4 channels separately
3. Use the merge.ipynb code in the preprocess folder to merge the images into different channels
4. Use the copy.ipynb code in the preprocess folder to copy the images from the initial folder to the model input data folder, and extract the test set at a rate of 20%

Model Run Code
train_env2s_3classes_20230915.ipynb、train_env2s_angii_20230918.ipynb、train_env2s_iso_20230919.ipynb


preprocess/             # preprocessing steps
- classify.ipynb   # Channel classification
- crop.ipynb       # Image cropping
- merge.ipynb      # Channel merging
- copy.ipynb       # Copy and dataset division

models/                 # Model framework
- model_efficientnetv2_1ch.py    # 1 channel model
- model_efficientnetv2_3ch.py    # 3 channel model
- model_efficientnetv2_4ch.py    # 4 channel model

utils/                  # Functional modules
- my_dataset.py    # Custom dataset classes
- utils_env2.py    # Functions in the training flow 

weights/                # Store the weights of the pre-training
- pre_efficientnetv2-s.pth       


train_env2s_3classes_20230915.ipynb  # Model calls are made for the three-classification task
train_env2s_angii_20230918.ipynb      # AngII specific model
train_env2s_iso_20230919.ipynb        # ISO-specific model
