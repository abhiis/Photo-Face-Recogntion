import cv2
import face_recognition
import os
import pickle
import shutil


def encode_faces(directory):
    known_faces = []
    known_names = []
    for person in os.listdir(directory):
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


def tag_faces_in_image(image_path, known_faces, known_names, output_folder, image_name):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Check if there is a match
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Export the identified face
            face_image = image
            person_folder = os.path.join(output_folder, name)
            os.makedirs(person_folder, exist_ok=True)
            face_filename = os.path.join(
                person_folder, f"face_{match_index} {image_name}.jpg")
            cv2.imwrite(face_filename, face_image)

        # Draw a rectangle and label the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the tagged image
    # cv2.imshow("Tagged Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to the directory containing all the images to be processed
image_directory = r"D:\Weddings\Anurag Wedding\EVENT 01"

# Path to the directory where the identified faces will be exported
output_directory = r"C:\Users\abhis\Desktop\The Big Grill\Photo Face Recogntion\Processed"

# Path to the pickled face encodings and names
encodings_file = "known_faces.pkl"
names_file = "known_names.pkl"

# Load the saved face encodings and names
with open(encodings_file, "rb") as f:
    known_faces = pickle.load(f)
with open(names_file, "rb") as f:
    known_names = pickle.load(f)

# Process all images in the directory and export the identified faces
for image_name in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_name)
    tag_faces_in_image(image_path, known_faces, known_names,
                       output_directory, image_name)
