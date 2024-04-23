import tensorflow as tf  # Importing TensorFlow for model operations
import streamlit as st  # Importing Streamlit for the web-based interface
import numpy as np  # Importing NumPy for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for visualization
from PIL import Image  # Importing PIL to handle image operations

def load_model(model_path):
    # Load a pre-trained Keras model from the specified path without recompiling it
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image, target_size):
    # Load and preprocess the image for model input
    # Resize the image to the specified target size
    image = tf.keras.utils.load_img(image, target_size=target_size)
    # Convert the image to a numpy array
    image = tf.keras.utils.img_to_array(image)
    # Add an additional dimension to match the expected input shape (batch size)
    image = np.expand_dims(image, axis=0)
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    return image

def main():
    # Create a header for the Streamlit app
    st.header("Image Classification Model")

    # Define the path to the model
    model_path = "D:\\Google drive\\Fruits_Vegetables\\Image_Recognition.keras"
    # Load the model using the specified path
    model = load_model(model_path)

    # Define the list of data categories that the model can classify
    data_categories = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
        'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
        'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
        'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]

    # Create a file uploader in the Streamlit app
    uploaded_files = st.file_uploader("Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Check if any files have been uploaded
    if uploaded_files:
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            try:
                # Display the name of the uploaded image
                st.write(f"### Image: {uploaded_file.name}")

                # Open and display the original image
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Original Image", use_column_width=True)

                # Preprocess the image to the required input format
                resized_image = preprocess_image(uploaded_file, target_size=(180, 180))

                # Predict the class of the image using the loaded model
                st.set_option('deprecation.showPyplotGlobalUse', False)  # Suppress deprecation warning
                prediction = model.predict(resized_image)
                predicted_class_index = np.argmax(prediction)  # Find the index of the class with the highest probability
                confidence = np.max(prediction) * 100  # Get the highest confidence level
                predicted_class = data_categories[predicted_class_index]  # Get the class name based on the index

                # Display the predicted class and confidence level
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                # Create a bar plot to visualize the prediction probabilities
                plt.bar(data_categories, prediction[0])
                plt.xlabel('Fruit/Vegetable Class')  # Label for the x-axis
                plt.ylabel('Probability')  # Label for the y-axis
                plt.title('Prediction Probabilities')  # Title for the plot
                plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

                # Display the plot in the Streamlit app
                st.pyplot()

                st.write("---")  # Separator for each image
            except Exception as e:
                # Display an error message if an exception occurs
                st.error(f"Error processing {uploaded_file.name}: {e}")

    # Create a "Reset" button to clear the uploaded files
    if st.button("Reset"):
        uploaded_files.clear()  # Clear the list of uploaded files

# Entry point for the script when executed
if __name__ == "__main__":
    main()  # Execute the main function
