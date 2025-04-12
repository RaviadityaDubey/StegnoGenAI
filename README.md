
# StegoGenAI - Secure Message Hiding and Steganalysis Using Generative AI

## Overview
StegoGenAI is a project that combines LSB (Least Significant Bit) steganography for hiding secret messages in images and deep learning-based steganalysis for detecting hidden messages. The system uses:
- **Streamlit** for a user-friendly web interface.
- **PyTorch** for training a Convolutional Neural Network (CNN) model to detect stego images.
- **LSB steganography** technique to embed secret messages in images.

## Features
- **Hide Message**: Users can hide a secret message in any uploaded image.
- **Reveal Message**: Users can extract hidden messages from stego images.
- **Detect Stego Image**: Classify images as **Cover** or **Stego** using a CNN model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RaviadityaDubey/SteganooGen.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Model Training
The CNN model can be trained on new datasets using `cnn_model.py`. Ensure that the dataset is placed in the `dataset/cover` and `dataset/stego` folders.

## Usage
1. **Hide Message**: Upload an image and a message to hide.
2. **Reveal Message**: Upload a stego image to extract the hidden message.
3. **Detect Stego**: Upload an image to classify it as a Cover or Stego.

## License
MIT License
