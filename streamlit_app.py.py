import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
# Make sure the model file 'image_classification_model.h5' is in the same directory or provide the correct path.
try:
    model = load_model('image_classification_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

st.title('Exam Cheating Detection')

if model is not None:
    st.write("Model loaded successfully!")
    # Add file uploader
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((128, 128))  # Resize to target size
        image_array = np.array(image)     # Convert to NumPy array
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
        # Normalize pixel values (assuming training data was scaled by 1/255)
        processed_image = image_array / 255.0

        st.write("Image preprocessed.")

        # Make prediction
        prediction = model.predict(processed_image)

        # Interpret and display the prediction
        # Assuming your model has trained with classes corresponding to indices
        # You might need to map indices to actual class names (e.g., ['cheating', 'not_cheating'])
        # For this example, let's assume class 0 is 'Not Cheating' and class 1 is 'Cheating'
        # Based on the training output, it seems there might be only one class.
        # We need to handle the case where the model was trained on a single class or multiple classes.

        # If the model was trained on a single class and outputs a single value (e.g., using sigmoid for binary)
        # Or if it's a single class with softmax (which will output [1.0] for that class)
        if prediction.shape[-1] == 1:
             # Assuming a single class model, the prediction value itself can be interpreted
             # This might not be a typical setup for 'cheating' vs 'not_cheating'
             # Let's assume for now that a high value indicates the presence of the single class
             # and we need a threshold. However, based on the training output (accuracy 1.0000, loss 0.0000e+00)
             # and the model summary showing a single output neuron with softmax, it seems the model
             # was trained on a dataset with only one class. In this scenario, the model will always predict
             # that single class with a confidence of 1.0.

            predicted_class_index = np.argmax(prediction) # This will always be 0 for shape (batch_size, 1)
            confidence_score = prediction[0][predicted_class_index]

            # Since the training data had only one class, let's assume that class is 'Cheating' for the purpose of demonstration.
            # In a real-world scenario with 'cheating' and 'not_cheating', you would have 2 classes.
            class_names = ['Cheating'] # Replace with your actual class names if you have more than one

            predicted_class_name = class_names[predicted_class_index]


            st.write(f"Prediction: {predicted_class_name}")
            st.write(f"Confidence: {confidence_score:.4f}")


        elif prediction.shape[-1] > 1:
            # For multi-class classification (e.g., 'cheating', 'not_cheating')
            predicted_class_index = np.argmax(prediction)
            confidence_score = prediction[0][predicted_class_index]

            # You would need to define your class names based on your training data
            # Example: class_names = ['Not Cheating', 'Cheating']
            # For demonstration, let's use dummy class names if the model outputs more than one value
            # (which is unlikely based on the training logs).
            # If the model was intended for binary classification, the output layer should have 2 units with softmax,
            # or 1 unit with sigmoid. The current model summary shows 1 unit with softmax, which is unusual for binary.
            # Assuming the model was intended for binary classification with a single softmax output:
            # Let's interpret the single output as the probability of the positive class ('Cheating').
            # This contradicts the softmax activation on a single neuron.
            # Given the model summary and training output, it strongly suggests a single-class dataset.
            # However, if it were intended for binary, a single sigmoid output would be more standard.
            # Let's proceed assuming a single class output as observed.

            # If, hypothetically, there were two classes and the output was [prob_class0, prob_class1]
            # predicted_class_index = np.argmax(prediction)
            # confidence_score = prediction[0][predicted_class_index]
            # class_names = ['Class 0', 'Class 1'] # Replace with actual class names
            # predicted_class_name = class_names[predicted_class_index]
            # st.write(f"Prediction: {predicted_class_name}")
            # st.write(f"Confidence: {confidence_score:.4f}")

            # Since the model summary indicates a single output neuron with softmax, and the training logs show
            # 1 class found, we will stick to the single-class interpretation for the prediction display.
             predicted_class_index = np.argmax(prediction) # This will always be 0
             confidence_score = prediction[0][predicted_class_index]
             class_names = ['Cheating'] # Assuming the single class is 'Cheating'
             predicted_class_name = class_names[predicted_class_index]

             st.write(f"Prediction: {predicted_class_name}")
             st.write(f"Confidence: {confidence_score:.4f}")


else:
    st.write("Model not loaded. Please check the model file.")
