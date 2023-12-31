# Road-Sign-identification

## Dataset Preparation
1. Download dataset here https://drive.google.com/file/d/1kE5CocAu1qT0ENl2HOoVk9d7LE17SnuV/view?usp=sharing
2. Ensure your dataset is in the standard format with images and corresponding annotations in the `img` and `annotation.txt` format.
3. If your dataset is not in the standard format, perform data augmentation.
4. Split and save the dataset into `train` and `test` folders.

## Run the data preprocessing notebook:
```shell
  jupyter notebook 01_data_preprocessing.ipynb
```
## YOLOv5 Training
1. Run the object_detection_yolo_training.ipynb notebook for training the YOLOv5 model.

2. Create a data.yaml file with the following specifications:

a. train and test folder paths
b. Number of categories
c. Class name list containing class names

3. Install requirements:
```bash
pip install -r requirements.txt
```
4. Train the model
```bash
!python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 20
```
+ Arguments:
  +  --data: Specify the path to the data.yaml file.
  +  --cfg: Model checkpoint.
  +  --batch size: Batch size.
  +  --name: Path to the folder used to save training logs/weights.
  +  --epochs: Number of epochs.

5. Save the ONNX model
```bash
!python export.py --weights runs/train/Model9/weights/best.pt --include onnx --simplify --opset 12
```
+ Arguments:
  +  --weights: Path to the weight file.
  +  --include: ONNX format.
  +  --simplify: Simplify model to reduce complexity for deployment.
  +  --opset: Version of ONNX to export.

Download saved model: https://drive.google.com/file/d/18h2dT3Z5mXt4ClVDnYBzmo19LhYsVn9N/view?usp=sharing

## Run yolo_predictions.py and add the path of the saved model.
## Result
+ ![Prediction sample](https://github.com/ChetanB1997/Road-Sign-identification/blob/main/image2.jpg)
