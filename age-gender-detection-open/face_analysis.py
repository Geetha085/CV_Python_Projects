import cv2
import matplotlib.pyplot as plt

# File paths
image_path = 'F:/MCA/Python Project/demo.jpeg'
face_model = 'opencv_face_detector_uint8.pb'
face_proto = 'opencv_face_detector.pbtxt'
age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'

# Load models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

image = cv2.resize(image, (720, 640))
h, w = image.shape[:2]

# Detect face
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], swapRB=False)
face_net.setInput(blob)
detections = face_net.forward()

# Loop through detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.7:
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender prediction
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Age prediction
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show result
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Face Analysis Output")
plt.show()
