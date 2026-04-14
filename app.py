import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
from PIL import Image

# 1. The EMNIST Balanced Mapping (47 classes)
class_mapping = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "d", 39: "e", 40: "f", 41: "g", 42: "h", 43: "n", 44: "q", 
    45: "r", 46: "t"
}

# 2. Load the "brain" - Wrapped in a try-except for safety
try:
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    knn = joblib.load('knn_model.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.title("🖊️ EMNIST Handwriting Recognizer")
st.write("Draw a character in the box, then click Predict!")

# 3. Create a drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=25,        # Optimized thickness
    stroke_color="#FFFFFF", # White ink
    background_color="#000000", # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 4. Processing Logic
if canvas_result.image_data is not None:
    # Convert canvas to grayscale
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    # Resize to 28x28
    img_resized = img.resize((28, 28))
    
    # --- IMPORTANT: EMNIST TRANSPOSE FIX ---
    # Most EMNIST models expect the image to be transposed/flipped
    img_final = img_resized.transpose(Image.TRANSPOSE)
    
    # Preview for the user (to see what the model sees)
    if st.checkbox('Show Model Input Preview'):
        st.image(img_final, width=100, caption="28x28 Input")

    if st.button('Predict'):
        # Step A: Flatten and reshape
        flat_img = np.array(img_final).flatten().reshape(1, -1)
        
        # Step B: Apply Scaler (Must be the same one from training)
        scaled_img = scaler.transform(flat_img)
        
        # Step C: Apply PCA
        pca_img = pca.transform(scaled_img)
        
        # Step D: KNN Prediction
        numeric_prediction = knn.predict(pca_img)[0]
        
        # Step E: Look up label
        character_prediction = class_mapping.get(numeric_prediction, "Unknown")
        
        # Display results with style
        st.divider()
        st.header(f"Result: **{character_prediction}**")
        st.write(f"Class ID: {numeric_prediction}")
        
        if character_prediction in ["0", "O", "1", "I", "L", "l"]:
            st.warning("Note: These characters are structurally similar and common points of failure for KNN.")