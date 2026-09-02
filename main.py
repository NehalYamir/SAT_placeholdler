import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image


def get_image_diff(imgA_path, imgB_path, min_contour_area=100):
    """
    Computes the difference between two images and returns the contoured output.
    """
    imgA = cv2.imread(imgA_path)
    imgB = cv2.imread(imgB_path)

    if imgA is None or imgB is None:
        return None

    # Resize if dimensions do not match
    if imgA.shape != imgB.shape:
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    # Convert to grayscale for processing
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurA = cv2.GaussianBlur(grayA, (5, 5), 0)
    blurB = cv2.GaussianBlur(grayB, (5, 5), 0)

    # Compute absolute difference and apply threshold
    diff = cv2.absdiff(blurA, blurB)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the "after" image
    output_img = imgB.copy()
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Draw in red (BGR format here: 0, 0, 255)
            cv2.drawContours(output_img, [contour], -1, (0, 0, 255), 2)

    # Convert from OpenCV BGR to Web-friendly RGB
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    return output_rgb


# ==========================================
# Streamlit Web Interface
# ==========================================
st.set_page_config(layout="wide", page_title="LEVIR Change Detection")

st.title("Satellite Image Change Detection Viewer")

# Define folder paths
DIR_A = "levir/A"
DIR_B = "levir/B"

# Check if directories exist
if not os.path.exists(DIR_A) or not os.path.exists(DIR_B):
    st.error(
        f"Please ensure the folders '{DIR_A}' and '{DIR_B}' exist in the same directory as this script."
    )
else:
    # Get list of images
    valid_extensions = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(DIR_A) if f.lower().endswith(valid_extensions)]

    if not files:
        st.warning("No images found in the A folder.")
    else:
        # 1. Dropdown for user to select a specific image
        selected_file = st.selectbox("Select an image to analyze:", files)

        path_A = os.path.join(DIR_A, selected_file)
        path_B = os.path.join(DIR_B, selected_file)

        if not os.path.exists(path_B):
            st.error(f"Matching file '{selected_file}' not found in Folder B.")
        else:
            # 2. Add a slider to let the user tune the contour sensitivity live
            min_area = st.slider(
                "Minimum Contour Area (filters out small noise)",
                min_value=0,
                max_value=500,
                value=100,
            )

            # 3. Process the images
            with st.spinner("Processing differences..."):
                output_image = get_image_diff(path_A, path_B, min_contour_area=min_area)

            if output_image is not None:
                # Convert raw BGR images to RGB for display
                img_A_disp = cv2.cvtColor(cv2.imread(path_A), cv2.COLOR_BGR2RGB)
                img_B_disp = cv2.cvtColor(cv2.imread(path_B), cv2.COLOR_BGR2RGB)

               # 4. Display in three columns side-by-side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.header("Image A (Before)")
                    st.image(img_A_disp, use_container_width=True)
                    
                with col2:
                    st.header("Image B (After)")
                    st.image(img_B_disp, use_container_width=True)
                    
                with col3:
                    st.header("Output (Changes)")
                    st.image(output_image, use_container_width=True)
            else:
                st.error("Error processing the images. Please check the files.")
