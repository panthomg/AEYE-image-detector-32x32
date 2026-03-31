# 🤖 AEye: Deep Learning Image Detector
A custom-trained Neural Network designed to distinguish between real-world photography and AI-generated imagery.

### 🌟 A Note from the Developer
I had a lot of fun making this, and I'm proud to successfully custom train an AI model for my needs. You too can train an AI model with ease with decent hardware and self-interest. I learned this myself by searching and asking help from chatbots to help me achieve this. I learned a lot in this journey and would love to share it with you guys. *Some of the technical explanations below were refined using AI for clarity and efficiency.*
### To Learn how things work in-depth
 To Learn how things work in-depth, what I learned, thought of and was curious about. Go to Repo's Wiki for more learnings!
---

## 🧠 How It Works (The Theory)
If you are new to AI, here is the "simple" version of what is happening inside this code:

1. **Digital Fingerprints:** AI generators (like Stable Diffusion) don't create images the way humans take photos. They start with random "noise" and clean it up. This leaves behind mathematical patterns called **Artifacts** that are invisible to humans but obvious to a computer.
2. **Convolutional Neural Networks (CNN):** We use an architecture called **EfficientNet-B0**. Think of it as a set of digital eyes that scan the image for these artifacts.
3. **Transfer Learning:** We didn't start from zero. We used a "Pre-trained" model from Google that already knows what shapes and colors look like. We just "re-educated" the final layer to specifically look for AI fingerprints.
4. **Tensors:** The computer doesn't see "pixels"; it sees a massive grid of numbers called a **Tensor**. Our model does high-speed calculus on these numbers to find the answer.

---

## 🚀 Step-by-Step Tutorial (For Beginners)

If you want to recreate this project on your own laptop, follow these steps:
## 📥 Quick Start

Clone the repository and move into the project folder:

```bash
git clone https://github.com/panthomg/AEYE-image-detector-32x32.git
cd AEYE-image-detector-32x32
```

### 1. The Environment
You need to get your hardware ready to talk to your code.
- Install **Python 3.11**.
- Install **VS Code**.
- Install the **CUDA-enabled PyTorch** (This allows your NVIDIA GPU to do the heavy lifting).
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. The Data (The AI's Textbook)
AI needs examples to learn. I used the **CIFAKE Dataset** from Kaggle, which contains 100,000 images labeled "REAL" or "FAKE." 
- Organize your folders: `/train/REAL`, `/train/FAKE`, `/test/REAL`, `/test/FAKE`.

### 3. The Training
Run the `train.py` script.
- **Epochs:** Each "Epoch" is one full pass through the dataset.
- **Loss:** This is the AI's "penalty score." The lower the number, the smarter the AI is becoming. 
- **Wait Time:** On an RTX 4050, this takes about 45 minutes for 5 epochs.

### 4. The App
Once the file `ai_detector.pth` appears in your folder, run `app.py`. This launches a web interface where you can drag and drop your own images!

---

## 🚧 Common Hurdles (Read if you get errors!)
During this project, I ran into several "boss fights." Here is how to beat them:
- **NumPy 2.0 Conflict:** Many AI libraries aren't ready for NumPy 2.0 yet. If your code crashes, downgrade: `pip install "numpy<2"`.
- **CUDA Not Found:** If your code says "Training on CPU" even though you have a GPU, you likely installed the wrong PyTorch. Use the `--index-url` command mentioned in the Setup.
- **Out of Memory (OOM):** If your GPU crashes, lower your `BATCH_SIZE` in the code from 32 to 16.

---

## 💻 Tech Stack & Hardware
- **Language:** Python 3.11
- **Framework:** PyTorch
- **Architecture:** EfficientNet-B0
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM) *This is the hardware I trained the model in, took about 40 mins to complete training*
- **Interface:** Gradio Web UI

## 📊 Results
- **CIFAKE Training:** Final Loss of **0.0572** (Approx. 97% Accuracy). 
```
Training started on cuda...
Epoch 1 finished. Loss: 0.1496
Epoch 2 finished. Loss: 0.0997
Epoch 3 finished. Loss: 0.0798
Epoch 4 finished. Loss: 0.0655
Epoch 5 finished. Loss: 0.0572
```

## 👤 About Me
I am a 17-year-old developer interested in Machine Learning and Computer Vision. This is my first time training a model on my own hardware by myself. This project was built to explore how hardware acceleration (CUDA) can be used to solve modern digital authentication challenges.
<img width="700" height="394" alt="Screenshot 2026-03-23 144255" src="https://github.com/user-attachments/assets/4da1eeec-7ac9-45d5-baa8-7d7048eb1aa4" />
<img width="700" height="394" alt="Screenshot 2026-03-23 144313" src="https://github.com/user-attachments/assets/3e91c45a-62e5-48b8-bf99-c7cb168027ff" />
<img width="700" height="394" alt="Screenshot 2026-03-23 150224" src="https://github.com/user-attachments/assets/1a6fa622-44f0-4e5f-88e9-c5597938a376" />
<img width="700" height="394" alt="Screenshot 2026-03-23 150230" src="https://github.com/user-attachments/assets/c9a25203-2f80-42fd-88ba-31cd78309e2e" />
<img width="700" height="394" alt="Screenshot 2026-03-23 151650" src="https://github.com/user-attachments/assets/e285ff3b-407d-46c1-9d24-d0f5a3377f87" />

---
