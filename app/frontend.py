import requests
import streamlit as st
import os
from PIL import Image

st.title("Water Bodies Segmentation")

uploading = st.container()
uploaded_file = uploading.file_uploader(
    label="Upload an image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False
)

# Path to the directory where you want to save the uploaded file
save_dir = "saved_images"
# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# ------------- Submit Button ---------------
submit = None
if uploaded_file:
    submitting = st.container()
    submit = submitting.button("Submit")
    st.markdown("""---""")

# -------------- Display uploaded file ------------
generating = st.container()
if submit:
    # Save the uploaded image to the 'saved_images' directory
    image_path = f"{save_dir}/{uploaded_file.name}"
    with open(image_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    # Make a request to the backend with the image path
    data = {"image_name": uploaded_file.name}
    
    # Send the image to the backend for segmentation
    response = requests.post("http://127.0.0.1:8000/store", json=data)

    if response.status_code == 200:
        result_path = response.json().get("result_path")

        # Display the original image and the segmented result
        st.subheader("Original Image")
        st.image(uploaded_file)

        st.subheader("Segmented Image")
        result_image = Image.open(result_path)
        st.image(result_image)
    else:
        st.error("Failed to segment the image.")
