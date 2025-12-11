# Driver Behavior Detection using CNNs (5 Classes)

This project implements three convolutional neural network architectures
(AlexNet-style, VGG-style, ResNet-34-style) to classify driver behavior
into five categories:
- Safe driving
- Talking on the phone
- Texting on the phone
- Turning
- Other activities

## 1. Requirements
Install dependencies:
pip install tensorflow numpy pandas pillow opencv-python matplotlib scikit-image

## 2. Dataset Setup
Use the Revitsone 5-Class dataset, with the folder structure:
Revitsone-5classes/
    safe_driving/
    talking_phone/
    texting_phone/
    turning/
    other_activities/

Update dataset path in the notebook if running locally:
DATA_ROOT = "/path/to/Revitsone-5classes"

## 3. Running the Notebook
1. Open driver-behavior-detection-cnn.ipynb
2. Ensure DATA_ROOT is correct
3. Run all cells, which include:
   - Loading images and removing corrupted files
   - Creating train/val/test splits
   - Building ImageDataGenerators
   - Training AlexNet, VGG, and ResNet-34 models
   - Plotting accuracy and loss curves
4. (Optional) Save a model:
model.save("model_name.h5")

## 4. Training Configuration
- Image size: 240×240×3
- Batch size: 64
- Epochs: 20
- Optimizer: Adam (lr=0.001)
- Loss: binary cross-entropy
