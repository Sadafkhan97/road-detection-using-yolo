Road Condition and Turn Detection using YOLOv8
This project is an object detection model trained to classify road images into four categories: right turn, left turn, straight, and unexpected. The dataset has been structured for training using the YOLOv8 model.

Prerequisites
Ensure you have the necessary libraries installed. This project uses:

Python 3.8+
YOLOv8 by Ultralytics
Required libraries (unzip, pip, and ultralytics)
Installation
Unzip the Dataset: Download the dataset ZIP file and extract it using the following commands:

bash
Copy code
!apt-get install unzip
!unzip "/content/Project Road.v4i.yolov11.zip" -d /content/extract/
!ls /content/extract/
This will unzip the dataset into the /content/extract/ directory.

Install YOLOv8: Install the YOLOv8 library by Ultralytics:

bash
Copy code
pip install ultralytics
Import YOLO: Import YOLO and load a pre-trained small model (yolov8s.pt):

python
Copy code
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Pre-trained small model
Training the Model
Train Command: Run the following command to train the YOLOv8 model on the dataset:

bash
Copy code
!yolo task=detect mode=train model=yolov8s.pt data=/content/extract/data.yaml epochs=50 batch=16 imgsz=224 lr0=0.01 lrf=0.1 patience=20 plots=True
task=detect: Specifies the task type as detection.
mode=train: Sets the mode to training.
model=yolov8s.pt: Uses the YOLOv8 small pre-trained model as a starting point.
data=/content/extract/data.yaml: Points to the dataset configuration file.
epochs=50: Trains the model for 50 epochs.
batch=16: Uses a batch size of 16.
imgsz=224: Sets image size to 224x224 pixels.
lr0=0.01, lrf=0.1: Sets the initial and final learning rates.
patience=20: Early stopping patience.
plots=True: Generates and saves plot results.
Check Training Outputs: To view the output files from training:

bash
Copy code
!ls runs/detect/train
Validation
Run Validation: Use the following command to validate the model on the dataset:

bash
Copy code
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=/content/extract/data.yaml
task=detect: Specifies the detection task.
mode=val: Sets the mode to validation.
model=runs/detect/train/weights/best.pt: Loads the best model from training.
data=/content/extract/data.yaml: Uses the dataset configuration file.
Check Validation Outputs: List the outputs from validation:

bash
Copy code
!ls runs/detect/val
Visualization of Results
To display the confusion matrix and additional results:

Display Confusion Matrix:

python
Copy code
from IPython.display import Image, display
confusion_matrix_path = 'runs/detect/train/confusion_matrix.png'
display(Image(filename=confusion_matrix_path, width=600))
Display Results:

python
Copy code
Image(filename='runs/detect/train/results.png', width=600)
Display Sample Labeling:

python
Copy code
Image(filename='/content/runs/detect/train/val_batch1_labels.jpg', width=600)
