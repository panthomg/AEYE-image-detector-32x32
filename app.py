import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Load the Brain
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("ai_detector.pth"))
model.eval()

# 2. Prediction Function
def predict_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    return {"Fake (AI)": float(probs[0]), "Real": float(probs[1])}

# 3. Create Interface
interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(),
    title="AI vs Real Image Detector"
)

interface.launch()
