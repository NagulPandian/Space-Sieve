#!/usr/bin/env python
# coding: utf-8

# In[ ]

import os


# Specify the path of the directory you want to change to
path = "C:/Applied_Machine_Learning/Project/yolov5"

# Change the current working directory to the specified directory
os.chdir(path)


import streamlit as st
import tensorflow as tf
import cv2
import time
import numpy as np

from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from pathlib import Path
import shutil

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.plots import save_one_box
import torch

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image


# Center-align the title using columns
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("SPACE SIEVE")
    

def resize_image(image, size=(320, 320)):
    """ Resize image to given size. """
    return cv2.resize(image, size)

def enhance_contrast(image, method='CLAHE'):
    """ Enhance contrast using specified method. """
    if method == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # Grayscale image
            return clahe.apply(image)
            
    elif method == 'hist_equal':
        if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(ycrcb)
            cv2.equalizeHist(channels[0], channels[0])
            cv2.merge(channels, ycrcb)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:  # Grayscale image
            return cv2.equalizeHist(image)
    else:
        return image


def denoise_image(image, method='non_local_means'):
    """ Denoise image using specified method. """
    if method == 'non_local_means':
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif method == 'gaussian_blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image


# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

    image_data = uploaded_file.read()

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Extract image name
    image_name = uploaded_file.name
    source = 'tests/' + str(image_name)
    
    img = resize_image(img) # resizing

    desired_height = 128
    desired_width = 128

    # Centering the image using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:  # Use the middle column
        st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Processing steps
    with st.spinner('Processing image...'):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        

        # Denoise
        status_text.text("Applying denoising algorithms...")

        img = denoise_image(img)
        
        time.sleep(1) 
        progress_bar.progress(30)
        

        # Contrast enhancement
        status_text.text("Enhancing image contrast...")

        img = enhance_contrast(img)
        
        processed_path = 'C:/Applied_Machine_Learning/Project/preprocessed'
        # Check if the destination folder is not empty
        if os.path.exists(processed_path):
            # Remove the entire directory and its contents
            shutil.rmtree(processed_path)
        save_dir = Path(processed_path)  # Update this path
        save_dir.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
        
        cv2.imwrite(processed_path + '/processed_img.jpg', img)
        
        time.sleep(1) 
        progress_bar.progress(50)

        

        # Get the list of files in the folder
        files = os.listdir(processed_path)
        # Filter out non-image files
        image_files = [file for file in files if file.lower().endswith(('.jpg'))]
        # Load the image
        processed_img_path = os.path.join(processed_path, image_files[0])
        # image = cv2.imread(image_path)

        
        # Object detection
        status_text.text("Detecting object in the image...")
        
        # Load the model
        weights_path = 'C:/Applied_Machine_Learning/Project/yolov5/runs/train/yolov5s_results4/weights/best.pt' # replace with your weights path
        device = select_device('') # select device ('cpu' or 'cuda:0')
        model = attempt_load(weights_path)  # If there's a device issue, we'll address it in the next line
        model.to(device)  # Ensure the model is on the correct device
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(320, s=stride)  # check image size
        
        def draw_square_box_in_folder(folder_path, box_color=(255, 0, 0), thickness=2, margin=100):
            # Get the list of files in the folder
            files = os.listdir(folder_path)
        
            # Filter out non-image files
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Ensure there's only one image file in the folder
            if len(image_files) != 1:
                print("Error: There should be exactly one image file in the folder.")
                return
            # Load the image
            image_path = os.path.join(folder_path, image_files[0])
            image = cv2.imread(image_path)
        
            # Make a copy of the original image to draw on
            image_with_box = image.copy()
            
            # Get the dimensions of the image
            height, width = image.shape[:2]
        
            # Calculate the coordinates for the square box
            x1 = margin  # Left edge
            y1 = margin  # Top edge
            x2 = width - margin  # Right edge
            y2 = height - margin  # Bottom edge
        
            # Draw the square box
            cv2.rectangle(image_with_box, (x1, y1), (x2, y2), box_color, thickness)
            # Overwrite the original image with the image with the box
            cv2.imwrite(image_path, image_with_box)
        
        dataset = LoadImages(processed_img_path, img_size=imgsz, stride=stride)
        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

        model.eval()
        for path, img, im0s, vid_cap, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
        
            # Inference
            pred = model(img, augment=False, visualize=False)[0]
        
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
        
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
        
                directory_path = 'C:/Applied_Machine_Learning/Project/detections'
                # Check if the destination folder is not empty
                if os.path.exists(directory_path):
                    # Remove the entire directory and its contents
                    shutil.rmtree(directory_path)
                
                save_dir = Path(directory_path)  # Update this path
                save_dir.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
        
                # Write results and save the cropped images with padding
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    file_name = save_dir / f"{label}.jpg"  # Define the file name
                    
                    save_one_box(xyxy, im0, file=file_name, pad=300, save=True)  # Adjust t
        
                    draw_square_box_in_folder(directory_path)
                    

        time.sleep(1) 
        progress_bar.progress(70)
        
        # Get the list of files in the folder
        files = os.listdir(directory_path)
        # Filter out non-image files
        image_files = [file for file in files if file.lower().endswith(('.jpg'))]
        # Load the image
        img_path = os.path.join(directory_path, image_files[0])
        
        # Normalize and pass to CNN for final classification
        status_text.text("Classifying the image...")
        
        # Load the model
        model = load_model('C:/Applied_Machine_Learning/Project/final_CNN_model.h5')
        model.summary()  # Optional: to verify that model is loaded correctly
        
        # Load the image and preprocess it
        img = image.load_img(img_path, target_size=(128, 128))  # Resize the image to match the model's expected input
        img_array = image.img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        img_array /= 255.0  # Normalize to [0,1]
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = (predictions > 0.5).astype(int)  # Use 0.5 as the threshold for binary classification

        # Conditional check for predicted class
        if predicted_class[0][0] == 0:
            pred_class = 'Debris'
        else:
            pred_class = 'Satellite'
        
        time.sleep(1) 
        progress_bar.progress(90)

        message = "Detected: " + str(pred_class)
        # Display the success message
        st.success(message)
            

        # Display results
        status_text.text("Processing complete!")
        time.sleep(2)
        progress_bar.progress(100)

