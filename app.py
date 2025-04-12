import streamlit as st
from PIL import Image
from stegano import lsb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

# ==== CNN Model ====
class StegoCNN(nn.Module):
    def __init__(self):
        super(StegoCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# Load trained model (replace with your own .pth file)
model = StegoCNN()
model.load_state_dict(torch.load("stegocnn.pth", map_location=torch.device("cpu")))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==== Streamlit Interface ====
st.set_page_config(page_title="StegoGen", layout="centered")
st.title("🕵️‍♂️ StegoGen - Steganography & Detection")

st.sidebar.header("Choose an Operation")
operation = st.sidebar.radio("Select", ["Hide Message", "Reveal Message", "Detect Stego Image"])

# ==== Hide Message ====
if operation == "Hide Message":
    st.subheader("📤 Hide Secret Message Inside Image")
    uploaded_image = st.file_uploader("Upload an Image (PNG)", type=["png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

        secret_msg = st.text_area("Enter the secret message:")
        if st.button("Encode"):
            if secret_msg:
                output_image = lsb.hide(uploaded_image, secret_msg)
                output_path = "stego_image.png"
                output_image.save(output_path)
                st.success("✅ Message hidden successfully!")

                with open(output_path, "rb") as f:
                    st.download_button("Download Stego Image", f, "stego_image.png", "image/png")
            else:
                st.error("❌ Please enter a secret message.")

# ==== Reveal Message ====
elif operation == "Reveal Message":
    st.subheader("📥 Reveal Hidden Message from Image")
    stego_img = st.file_uploader("Upload the Stego Image (PNG)", type=["png"])

    if stego_img:
        try:
            hidden_msg = lsb.reveal(stego_img)
            if hidden_msg:
                st.success("✅ Hidden Message:")
                st.code(hidden_msg)
            else:
                st.warning("⚠️ No hidden message found.")
        except:
            st.error("❌ Unable to decode. Ensure the image is valid.")

# ==== Detect Stego ====
elif operation == "Detect Stego Image":
    st.subheader("🔍 Steganalysis - Detect if an Image is Stego or Cover")
    detect_img = st.file_uploader("Upload an Image (PNG)", type=["png"])

    if detect_img:
        img = Image.open(detect_img).convert("RGB")
        st.image(img, caption="Image to Analyze", use_column_width=True)

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_tensor)
            class_idx = torch.argmax(pred).item()
            class_names = ["Cover", "Stego"]

            st.success(f"🔎 Prediction: **{class_names[class_idx]}**")
