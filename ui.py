import streamlit as st
from PIL import Image

def upload_image():
    """
    Display a Streamlit file uploader. If an image is uploaded, show a preview and return it.

    Returns:
        PIL.Image.Image or None: The uploaded image as a PIL Image (RGB), or None if nothing uploaded.
    """
    st.title("🛏️ Drowsiness Detection App")
    st.write("Upload a face image and the model will predict whether the person is drowsy (1) or not (0).")

    uploaded_file = st.file_uploader(
        label="Choose an image file (JPG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Convert the uploaded file to a PIL image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image

    return None
