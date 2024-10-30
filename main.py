import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("new_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(210,210))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    for _ in predictions:
        for val in _ :
            if val >= 0.5:
                return 1
            else:
                return 0


with st.sidebar:
    app_mode = option_menu(
        menu_title = "Dashboard",
        options = ["Home", "About", "COVID-19 Prediction"]
    )

#Main Page
if(app_mode=="Home"):
    st.header("COVID-19 Prediction Model")
    st.markdown("""
    Welcome to the COVID-19 Detection System!
    
    This project objective is to help in identifying COVID patients as earlier as possible so that the patient can be quarantined for controlling the spread of the disease.
    ### How It Works
    1. **Upload Image:** Go to the **COVID-19 prediction** page and upload an image of CT scan of lungs of the patient.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify it whether he/she is suffering from COVID-19 or not.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate COVID-19 detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **COVID-19 prediction** page in the sidebar to upload an image and check the results.

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo. 
                8439 Lung CT scans of patients infected by COVID-19 (SARS-CoV-2) and also suspicious ones with normal or non-COVID-19 results. Converted to 512x512px PNG images.
                This is a large public COVID-19 (SARS-CoV-2) lung CT scan dataset, containing total of 8,439 CT scans which consists of 7,495 positive cases (COVID-19 infection) and 944 negative ones (normal and non-COVID-19).
                """)

#Prediction Page
elif(app_mode=="COVID-19 Prediction"):
    st.header("COVID-19 Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:  # Check if an image is uploaded
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        with st.spinner("Please wait..."):
            result_index = model_prediction(test_image) # Display the image with full column width
            #Reading Labels
            class_name = ['positive', 'negative']
            st.success("The model is predicting that the patient is COVID-19 {}.".format(class_name[result_index]))