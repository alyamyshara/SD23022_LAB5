import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Page configuration
st.set_page_config(page_title="Image Classification with ResNet18", layout="centered")
st.title("Computer Vision Image Classification (ResNet18)")

# Step 2 & 3: Force CPU usage
device = torch.device("cpu")
st.write("Running on device:", device)

# Step 4: Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Step 5: Image preprocessing
weights = models.ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Load ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = pd.read_csv(labels_url, header=None)[0].tolist()

# Step 6: Upload image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 7: Convert image to tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Step 8: Inference and softmax
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability": float(top5_prob[i])
        })

    df = pd.DataFrame(results)

    st.subheader("Top-5 Predictions")
    st.table(df)

    # Step 9: Visualize probabilities
    st.subheader("Prediction Probability Chart")
    fig, ax = plt.subplots()
    ax.barh(df["Class"], df["Probability"])
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Image Classification Results")
    st.pyplot(fig)

# Step 10: Explanation
st.markdown("""
### Classification Process Explanation
1. User uploads an image.
2. Image is preprocessed using ResNet18 recommended transformations.
3. The image tensor is passed through the pretrained CNN model.
4. Softmax converts outputs into probabilities.
5. The top-5 predicted classes are displayed with confidence scores.
""")
