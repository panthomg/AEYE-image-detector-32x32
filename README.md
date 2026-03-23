# 🤖 AEye: Deep Learning Image Detector
A custom-trained Neural Network designed to distinguish between real-world photography and AI-generated imagery.

Before we start:
I had a lot of fun making this, and I'm proud to succesfully custom trian an AI model for my needs. You too can train an AI model with ease with a decent hardware and self interest. I learned this myself with searching and asking help from chatbots to help me achive this, I learned a lot of thing in this journey and would love to share it with you guys. *Some of the text below might be AI generated to save time and efficiency*


## 🚀 Overview
As AI image generators (like Midjourney and Stable Diffusion) become more realistic, the need for automated detection is critical. This project implements a **Convolutional Neural Network (CNN)** using **Transfer Learning** on the **EfficientNet-B0** architecture to detect microscopic "fingerprints" left by generative models.

## 💻 Tech Stack & Hardware
- **Language:** Python 3.11
- **Framework:** PyTorch
- **Architecture:** EfficientNet-B0 (Pre-trained on ImageNet)
- **Hardware used for Training:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- **Interface:** Gradio Web UI

## 📊 Methodology & Results
1. **Initial Training:** Trained on the **CIFAKE Dataset** (100k images).
   - Achieved a final loss of **0.0572** over 5 epochs.
   - High accuracy on low-resolution (32x32) "domain" images.
2. **Current Goal:** Fine-tuning on high-resolution (512x512) **Unsplash + Stable Diffusion XL** datasets to improve real-world generalization.

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/AIEye-Image-Detector.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training: `python train.py`
4. Launch the Web App: `python app.py`

## 👤 About Me
I am a 17-year-old developer interested in Machine Learning and Computer Vision. This project was built to explore how hardware acceleration (CUDA) can be used to solve modern digital authentication challenges.
