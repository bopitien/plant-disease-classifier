import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import zipfile
import io

# Define the custom PatchExtractor layer
from patch_extractor import PatchExtractor

def vit_classifier_page():
    st.title('🌿 ViT Plant Disease Classifier')

    # Custom CSS for Background and Fonts
    st.markdown("""
        <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Helvetica', sans-serif;
        }
        .stApp header {
            font-size: 2rem;
            color: #4CAF50;
            text-align: center;
        }
        .css-10trblm {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stFileUploader {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load the ViT model and class indices
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "vit_plant_disease_model.keras")
    
    # Ensure the custom layer (PatchExtractor) is recognized when loading the model
    with tf.keras.utils.custom_object_scope({'PatchExtractor': PatchExtractor}):
        model = tf.keras.models.load_model(model_path)
    
    class_indices_path = os.path.join(working_dir, "class_indices_vit.json")
    class_indices = json.load(open(class_indices_path))

    # Function to Load and Preprocess the Image
    def load_and_preprocess_image(image_path, target_size=(128, 128)):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array

    # Function to Predict the Class of an Image
    def predict_image_class(model, image_path, class_indices):
        try:
            preprocessed_img = load_and_preprocess_image(image_path)
            predictions = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_indices[str(predicted_class_index)]
            return predicted_class_name
        except Exception as e:
            return f"Error: {e}"

    # Function to handle multiple image predictions
    def predict_multiple_images(images):
        predictions = []
        for uploaded_image in images:
            image_name = uploaded_image.name
            prediction = predict_image_class(model, uploaded_image, class_indices)
            predictions.append({'Image': image_name, 'Prediction': prediction})
        return predictions

    st.markdown("""
        Welcome to the ViT Plant Disease Classifier! Upload an image of a plant leaf to identify possible diseases.
        
        **Instructions**:
        1. Click on the "Upload Image" button to upload a clear image of a plant leaf.
        2. Ensure that the image clearly shows the affected part of the leaf.
        3. After uploading, click the "Classify" button to see the prediction.
        
        *Supported file formats: jpg, jpeg, png.*
    """)

    # Single Image Upload Section
    st.markdown("### Upload a Single Image")
    uploaded_image = st.file_uploader("Upload a single image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image.resize((300, 300)), caption=f"Uploaded Image: {uploaded_image.name}", use_column_width=True)

        with col2:
            if st.button('Classify Image'):
                with st.spinner('Classifying...'):
                    prediction = predict_image_class(model, uploaded_image, class_indices)
                    st.success(f'Prediction: **{str(prediction)}**')

    # Multiple Image Upload Section
    st.markdown("### Upload Multiple Images")
    uploaded_images = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        col1, col2 = st.columns(2)
        with col1:
            for uploaded_image in uploaded_images:
                image = Image.open(uploaded_image)
                st.image(image.resize((150, 150)), caption=f"Uploaded: {uploaded_image.name}")
        with col2:
            if st.button('Classify All Images'):
                with st.spinner('Classifying...'):
                    results = predict_multiple_images(uploaded_images)
                    df_results = pd.DataFrame(results)
                    st.write(df_results)

                    # Option to download the results as CSV
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='image_predictions_vit.csv',
                        mime='text/csv',
                    )

    # Batch dataset processing (e.g., a zip file of images)
    st.markdown("### Upload a Zip of Images for Batch Prediction")
    uploaded_zip = st.file_uploader("Upload a zip file containing images...", type="zip")

    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall("temp_images")
            image_files = os.listdir("temp_images")

            if st.button('Classify Dataset'):
                predictions = []
                for image_file in image_files:
                    image_path = os.path.join("temp_images", image_file)
                    prediction = predict_image_class(model, image_path, class_indices)
                    predictions.append({'Image': image_file, 'Prediction': prediction})

                # Create a DataFrame for the results
                df_predictions = pd.DataFrame(predictions)
                st.write(df_predictions)

                # Provide a download link for CSV report
                csv_report = df_predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Dataset Predictions as CSV",
                    data=csv_report,
                    file_name='dataset_predictions_vit.csv',
                    mime='text/csv',
                )
