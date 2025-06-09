import streamlit as st
from ui import upload_image
from utils import load_model, predict

# -------------------------------
# 1) Set the path to your saved model file:
#    Change this to the correct path where you saved your .pth/.pt
# -------------------------------
MODEL_PATH = "./models/model.pth"  # ← replace with your actual path

# -------------------------------
# 2) Cache the model load so it isn't reloaded on every run:
# -------------------------------
@st.cache_resource
def get_model():
    """
    Load and cache the PyTorch model so that Streamlit does not reload it on every interaction.
    """
    model = load_model(MODEL_PATH)
    return model

# -------------------------------
# 3) Main Streamlit UI
# -------------------------------
def main():

    # apply the styles.css here
    with open("./styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load the model once
    model = get_model()

    # Let the user upload an image via ui.upload_image()
    image = upload_image()

    if image is not None:
        # Only show the “Predict” button if an image has been uploaded
        if st.button("Predict Drowsiness"):
            # Run inference
            label = predict(model, image)

            # Display results
            if label == 1:
                st.error("🚨 Drowsiness Detected (1)")
            else:
                st.success("✅ Not Drowsy (0)")

if __name__ == "__main__":
    main()
