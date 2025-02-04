import pandas as pd
import os
from PIL import Image
import shutil
import ast

# # Create the 'images' directories inside 'train' and 'val' if they don't already exist
# mkdir -p train/images
# mkdir -p val/images

# # Move all .jpg files to the 'images' directory under 'train'
# find train -maxdepth 1 -type f -name "*.jpg" -exec mv {} train/images/ \;

# # Move all .jpg files to the 'images' directory under 'val'
# find val -maxdepth 1 -type f -name "*.jpg" -exec mv {} val/images/ \;

# mkdir -p train/labels;
# mkdir -p val/labels;

# Load the Excel file
label_path = 'C:/Applied_Machine_Learning/Project/Data/master_data/label/final_data_label.csv'

label = pd.read_csv(label_path)
train_df = label[label['usage'] == 'train'][['filename', 'class', 'bbox']].sample(frac = 1, random_state = 32)
val_df = label[label['usage'] == 'validate'][['filename', 'class', 'bbox']].sample(frac = 1, random_state = 32)

# Directory where images are stored
train_image_folder = 'C:/Applied_Machine_Learning/Project/Data/yolo_data/train/images/'
val_image_folder = 'C:/Applied_Machine_Learning/Project/Data/yolo_data/val/images/'

train_label_folder = 'C:/Applied_Machine_Learning/Project/Data/yolo_data/train/labels/'
val_label_folder = 'C:/Applied_Machine_Learning/Project/Data/yolo_data/val/labels/'

def convert_boxes(df, image_folder, output_folder):
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        filename = row['filename']
        
        if isinstance(row['bbox'], str):
            bbox = ast.literal_eval(row['bbox'])
        else:
            bbox = row['bbox']
            
        y_min, x_min, y_max, x_max = bbox
        class_index = row['class']
        
        # y_min, x_min, y_max, x_max = row['y_min'], row['x_min'], row['y_max'], row['x_max']
        
        # Construct full image path
        image_path = os.path.join(image_folder, filename)
        
        # Try to open the image and handle the case where the image is not found
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Calculate normalized bounding box coordinates
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

                # Prepare label file content
                label_content = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"

                # Write to the corresponding label file
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(output_folder, label_filename)
                with open(label_path, 'a') as file:  # 'a' to append in case multiple boxes per image
                    file.write(label_content)

        except FileNotFoundError:
            print(f"Warning: {image_path} not found. Skipping.")
        except IOError:
            print(f"Error: Cannot open {image_path}. Skipping.")

# Example usage
convert_boxes(df = train_df, image_folder = train_image_folder, output_folder = train_label_folder)
convert_boxes(df = val_df, image_folder = val_image_folder, output_folder = val_label_folder)
