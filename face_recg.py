import cv2
import face_recognition
import os
import pickle


def encode_faces(directory):
    known_faces = []
    known_names = []
    for person in os.listdir(directory):
        print(f"Training: {person}")

        person_dir = os.path.join(directory, person)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                known_faces.extend(face_encodings)
                known_names.extend([person] * len(face_encodings))
    return known_faces, known_names


dataset_directory = r"C:\Users\abhis\Desktop\The Big Grill\Photo Face Recogntion\Persons"
known_faces, known_names = encode_faces(dataset_directory)

# Save the face encodings and corresponding names
with open("known_faces.pkl", "wb") as f:
    pickle.dump(known_faces, f)
with open("known_names.pkl", "wb") as f:
    pickle.dump(known_names, f)
