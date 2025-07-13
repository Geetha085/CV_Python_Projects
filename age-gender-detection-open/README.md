# Face Age & Gender Detection using OpenCV DNN

This project detects faces in an image and predicts their age and gender using OpenCV's deep learning module (DNN) and pre-trained models.

## 🔍 Features
- Face detection
- Age prediction
- Gender prediction
- Visualization using matplotlib

## 📂 Folder Structure
- `face_analysis.py`: Main script for face analysis
- `models/`: Contains all pre-trained models required for the project
- `images/`: Sample input images for testing

## 🛠️ Requirements
- Python 3.6+
- OpenCV
- matplotlib

## 📥 Download Model Files

Due to the large size of the model files, you need to manually download them and add them to the project. You can download the model files from the following links:

- [Download age_net.caffemodel](<https://drive.google.com/file/d/1-eo1wPH1Tw_txIVTt_hH61bHQqhv2QmU/view?usp=sharing>)
- [Download gender_net.caffemodel](<https://drive.google.com/file/d/1XJ0M-AkMHrchFTaQ8wJ7L7Ns8o7MdpfT/view?usp=sharing>)

### How to use the model files:

1. **Download the model files** from the provided links.
2. **Move the downloaded files** (`age_net.caffemodel` and `gender_net.caffemodel`) into the `models/` folder in your project directory.

Once the models are placed in the `models/` folder, the script will automatically load them for face detection and age/gender prediction.


To install the required libraries, use the following command:
```bash
pip install opencv-python matplotlib
