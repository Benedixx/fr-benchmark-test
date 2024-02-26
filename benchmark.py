import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras_facenet import FaceNet
import json

ghostnet_model_path = 'ghostnet_face/GhostFaceNet_W1.3_S1_ArcFace.h5'
facenet_model = FaceNet()

detector = cv2.CascadeClassifier('haar_cascade/haarcascade_frontalface_default.xml')

db_path = 'db'
test_path = 'test'

result = []


def prepare_db_embeddings(model):
    local_known_faces = {}

    for people in os.listdir(db_path):
        person_folder = os.path.join(db_path, people)
        encoding_list = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = preprocessing(img)

                if img is not None:
                    emb_img = get_embeddings(img, model)
                    encoding_list.append(emb_img.reshape(512,))

        local_known_faces[people] = encoding_list
    return local_known_faces


def prepare_test_set():
    test_images = []
    label_images = []

    for person in os.listdir(test_path):
        person_path = os.path.join(test_path, person)
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            label_images.append(person)
            img = cv2.imread(img_path)

            if img is not None:
                img = preprocessing(img)

                if img is not None:
                    test_images.append(img)

    return test_images, label_images


def preprocessing(image):
    faces = detector.detectMultiScale(image, 1.1, 4)

    if len(faces) == 0:
        return None

    x1, y1, width, height = faces[0]
    face = image[y1:y1 + height, x1:x1 + width]
    image = cv2.resize(face, (112, 112))
    image = (image - 127.5) * 0.0078125
    image = np.expand_dims(image, axis=0)
    return image


def get_embeddings(image, model):
    if model == facenet_model:
        embeddings = facenet_model.embeddings(image)
    else:
        embeddings = model.predict(image)
        embeddings = np.array(embeddings).astype("float32")[0]
    return embeddings


def calculate_similarity(emb_image, emb_img):
    similarity = np.dot(emb_image, emb_img)
    return similarity


def benchmark_accuracy(model, model_name):
    known_faces = prepare_db_embeddings(model)
    test_images, label_images = prepare_test_set()
    total_images = len(test_images)
    correct_predictions = 0

    for test_image, true_label in zip(test_images, label_images):
        detection_list = []

        for person_name, embeddings in known_faces.items():
            for emb_img in embeddings:
                similarity = calculate_similarity(get_embeddings(test_image, model), emb_img)
                temp_dict = {'person_name': person_name, 'similarity': similarity}
                detection_list.append(temp_dict)

        if detection_list:
            highest_similarity = max(detection_list, key=lambda x: x['similarity'])
            detected_person = highest_similarity['person_name']

            if detected_person == true_label:
                correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100
    print(f"Accuracy ({model_name}): {accuracy}%")

    # Store accuracy in JSON file
    accuracy_data = {
        'model': model_name,
        'accuracy': accuracy
    }
    result.append(accuracy_data)


# Load the models
print("Loading models...")
ghostnet_model = load_model(ghostnet_model_path, compile=False)

# Run the benchmarks
print("Running benchmarks for ghostnet")
benchmark_accuracy(ghostnet_model, 'GhostNet')
print("Running benchmarks for facenet")
benchmark_accuracy(facenet_model, 'FaceNet')

with open('accuracy.json', 'w') as f:
    json.dump(result, f)

print(result)