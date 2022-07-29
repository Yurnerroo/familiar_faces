from os import listdir
from os.path import isfile, join

import face_recognition as fr
import numpy as np


def find_faces(img_path: str):
    """
    Method, that finds face on given picture, and then,
    finds the most similar face among images in data.

    PATH EXAMPLE: "D:\\your\\path\\picture.jpg"
    :param img_path: path to your picture.
    :return: which familiar faces where found.
    """
    known_face_names = [pic for pic in listdir(".\\data") if isfile(join(".\\data", pic))]
    known_face_encodings = []
    
    for image in known_face_names:
        known_face = fr.load_image_file(f".\\data\\{image}")
        known_face_encodings.append(fr.face_encodings(known_face)[0])

    user_img = fr.load_image_file(img_path)
    user_img_face_loc = fr.face_locations(user_img)
    user_img_enc = fr.face_encodings(user_img, user_img_face_loc)

    faces_names = []

    for face_encoding in user_img_enc:
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        faces_names.append(name.replace(".jpg", ""))

    for face in faces_names:
        if face == "Unknown":
            faces_names.remove(face)

    if len(faces_names) > 1:
        print(f"People found: {', '.join(faces_names)}.")
    elif len(faces_names) == 1:
        print(f"People found: {faces_names[0]}.")
    else:
        print("No familiar people found..")


if __name__ == "__main__":
    print("Welcome to familiar_faces module!")
    file_path = input("Type path to image:")
    find_faces(file_path)


