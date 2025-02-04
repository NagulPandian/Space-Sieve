from IPython.display import Image
import os

# # Specify the path of the folder you want to enter
# folder_path = 'C:/Applied_Machine_Learning/Project/yolov5'

# # Change the current working directory to the specified folder
# os.chdir(folder_path)

# # It's a good practice to verify your current working directory
# # print("Current working directory:", os.getcwd())


from yolovs5.models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages
from utils.plots import save_one_box
import torch
import cv2
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt


def intiate():
    
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

    print("Boxed image saved successfully.")



# Load the image
source = 'C:/Applied_Machine_Learning/Project/Data/master_data/test/image_00000_2021.jpg'  # replace with your image path

def detect_box(source):

    intiate()
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    model.eval()
    # for path, img, im0s, vid_cap in dataset:
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
            shutil.rmtree(directory_path)
            
            save_dir = Path(directory_path)  # Update this path
            save_dir.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
    
            # Write results and save the cropped images with padding
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                file_name = save_dir / f"{label}.jpg"  # Define the file name
                
                save_one_box(xyxy, im0, file=file_name, pad=300, save=True)  # Adjust t
    
                draw_square_box_in_folder(directory_path)


detect_box(source)
