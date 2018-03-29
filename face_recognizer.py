import cv2, os
import numpy as np
from PIL import Image
import os.path

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()


def get_trained_data():
    global cascadePath, faceCascade, recognizer

    def get_images_and_labels(path):
        global cascadePath, faceCascade, recognizer

        already_read_faces = []
        with open("already_read_faces", 'r') as already_read_faces_file:
            already_read_faces = already_read_faces_file.read()
            already_read_faces = already_read_faces.split("\n")
            for item in already_read_faces:
                print item
        # raw_input("holding")

        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        images = []
        labels = []
        for image_path in image_paths:
            image_pil = Image.open(image_path).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(os.path.split(image_path)[1].split(".")[0])
            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                cv2.waitKey(50)
        return images, labels

    path = './faces'

    if os.path.isfile("trained_model"):
        print "Loading pre-trained images"
        recognizer.load("trained_model")

    else:
        print "Training face dataset"
        images, labels = get_images_and_labels(path)

        recognizer.train(images, np.array(labels))
        recognizer.save("trained_model")


def supply_face():
    global cascadePath, faceCascade, recognizer

    cv2.namedWindow("Record faces")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("Record faces", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(33)

        if key == 1048586:  # Save face on pressing enter key
            recognize_face(frame)

        elif key == 1048603:  # exit on ESC
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyWindow("Record faces")


def recognize_face(face):
    global cascadePath, faceCascade, recognizer

    predict_image_pil = cv2.cvtColor( face, cv2.COLOR_RGB2GRAY )

    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    #print str(len(faces)) + " faces found"

    face = get_the_most_relevant_face(faces)

    if face is None:
        return ''
    
    x, y, w, h = [ v for v in face ]
    nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])

    if conf < 100.0:
        print "\n\nThis face is of : " + str(nbr_predicted)
        print "confidence = " + str(conf)
        print "\n\n"

    else:
        #print "A result that seems to be irrelevant has been hidden"
        pass



def get_the_most_relevant_face(faces):
    if len(faces) == 0:
        print "No faces found"
        return None
    
    elif len(faces) > 1:
        old_face_height = 0
        current_face_height = 0
        current_face = -1
        chosed_face = 0
        
        for face in faces:
            current_face += 1
            x, y, w, h = [ v for v in face ]
            current_face_height = y+h
            if current_face_height > old_face_height:
                old_face_height = current_face_height
                chosed_face = current_face

        face = faces[chosed_face]
        x, y, w, h = [ v for v in face ]

        if h < 170: #Either wrong face is detected or they are too far
            print "Please come closer"
            return None            
        
        elif h > 220: #Either wrong face is detected or they are too far
            print "Please move a little farther"
            return None

    else:
        face = faces[0]
        x, y, w, h = [ v for v in face ]
        
        if h < 170: #Either wrong face is detected or they are too far
            print "Please come closer"
            return None            

        elif h > 220: #Either wrong face is detected or they are too far
            print "Please move a little farther"
            return None            

    return face    

            
if __name__ == '__main__':
    get_trained_data()
    supply_face()
