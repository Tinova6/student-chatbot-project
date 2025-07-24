import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (Needs to match the trained model)
# Load the pre-trained ResNet50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Get the number of input features for the last fully connected layer
num_ftrs = resnet50.fc.in_features

# Replace the last fully connected layer with a new one that has the number of output classes you need
# You need to know the number of classes your model was trained on.
# Based on the training output, it seems there were 5 classes.
num_classes = 5 # **IMPORTANT: Update this based on your actual number of classes**
resnet50.fc = nn.Linear(num_ftrs, num_classes)

# Load the trained state dictionary
model_path = 'resnet50_exam_cheating.pth' # Make sure this matches the filename you saved
try:
    resnet50.load_state_dict(torch.load(model_path, map_location=device))
    resnet50 = resnet50.to(device)
    resnet50.eval() # Set the model to evaluation mode
    model_loaded = True
    st.write("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False


st.title('Exam Cheating Detection')

if model_loaded:
    # Define the image transformations (same as training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Add file uploader
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        try:
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # Add a batch dimension
            input_batch = input_batch.to(device)

            st.write("Image preprocessed.")

            # Make prediction
            with torch.no_grad():
                output = resnet50(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get the predicted class and confidence
            # You will need the class names from your training data
            # Assuming class names are available from the training phase
            # Replace with your actual class names in the correct order
            # Based on the training output, classes were: ['cheating', 'giving code', 'giving object', 'looking friend', 'normal act']
            class_names = ['cheating', 'giving code', 'giving object', 'looking friend', 'normal act'] # IMPORTANT: Ensure this matches your training classes

            predicted_class_index = torch.argmax(probabilities).item()
            predicted_class_name = class_names[predicted_class_index]
            confidence_score = probabilities[predicted_class_index].item()

            st.write(f"Prediction: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence_score:.4f}**")

        except Exception as e:
            st.error(f"Error processing image or making prediction: {e}")

else:
    st.write("Model not loaded. Please check the model file and the error message above.")


