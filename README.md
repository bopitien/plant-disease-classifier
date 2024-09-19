# Plant Disease Prediction Project
## Overview
This project is aimed at building an image-based plant disease prediction system using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). I use the PlantVillage dataset, which contains over 50,000 images of healthy and infected leaves of various crops. 
The goal is to help farmers identify diseases affecting crops using machine learning models, which can be integrated into smartphone-based tools.

## Motivation
The need for food production is increasing as the global population grows, but infectious diseases in crops can lead to significant yield losses particularly in developing regions. By 2050, food production must increase by approximately 70% to meet global demand. 
This project aims to leverage deep learning to help detect plant diseases, offering a valuable tool to assist farmers in identifying and mitigating crop issues early.

## Dataset
I used the PlantVillage Dataset, which contains expertly curated images of healthy and diseased plant leaves. The dataset contains:
- Over 50,000 images: representing various crops like apple, potato, tomato, etc.
- Focus on color images: as they capture detailed visual features necessary for accurate classification.
- Images of diseased and healthy leaves: to train the model for effective disease detection.
  
Link to the dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

# Project Workflow

## Data Processing:
- Focused on color images and performed data augmentation (e.g., rotation, flipping) to increase the dataset's diversity.
- Split the dataset into training and testing sets.

# Model Training:

## CNN Training:
- Achieved a validation accuracy of 92.80%.
- Model saved as version plant_disease_prediction_model_V3.h5.

## ViT Training:
- Achieved a validation accuracy of 91.00%.
- Model saved as version vit_plant_disease_model.h5.
  

# Evaluation:
- Both CNN and ViT models were evaluated based on their accuracy, with CNN slightly outperforming ViT.

  

# Model Deployment:

- Dockerization: Created a Dockerfile to containerize the application.
- Deployment on Streamlit Cloud: Successfully deployed the project as a web application using Streamlit without relying on Docker.


# Model Details
- CNN model was designed to extract spatial features from the images, which are critical for detecting disease patterns.
- Architecture: Consists of multiple convolutional and pooling layers, followed by fully connected layers for classification.
- Performance: 92.80% validation accuracy.


- Vision Transformer (ViT) model utilizes transformer architecture to capture long-range dependencies in the image data.
- Architecture: Uses patch embeddings to convert images into sequences for the transformer to process.
- Performance: 91.00% validation accuracy.


## Technology Stack

-Python: Primary programming language.
-TensorFlow: For building and training the deep learning models.
-Streamlit: For deploying the web app interface.
-Docker: Used to containerize the application for consistent deployment.
-GitHub: For version control and collaboration.



## Future Improvements

- Expand the dataset: Include more crops and disease types for a broader application.
- Improve model accuracy: Fine-tune hyperparameters or explore more advanced architectures to achieve higher accuracy.
- Mobile integration: Develop a mobile app that integrates the model for real-time disease detection using smartphone cameras.
- 
## Conclusion
This project demonstrates how deep learning models like CNNs and ViTs can effectively identify plant diseases, offering a tool that could potentially assist farmers and help mitigate global food shortages.
The deployed web application can serve as a prototype for future mobile-based disease detection systems.
