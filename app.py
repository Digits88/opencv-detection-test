import cv2
import logging, coloredlogs

face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_smile.xml')


def detect_n_plot(frame, detect_eyes=True, detect_smile=True, logger=None):
    """
    @Parameters:
        frame: Original frame
        detect_eyes: Whether detect eyes or not
        detect_smile: Whether detect smiles or not
        logger: for logging
    @Returns:
        Frame with boxes surrounding faces and eyes
    """

    # Convert frames to grayscale images because the 
    # cascade classifier fits matrices with a single
    # color channel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_8U
    )

    # Plot a rectangle around each face
    for (face_x, face_y, face_w, face_h) in faces:

        if logger:
            logger.info("Face detected at coordinates ({}, {})".format(face_x, face_y))

        cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

        cv2.putText(frame, "Face", (face_x, face_y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

        # Take the face Region-of-Interest
        roi_frame = frame[face_y : face_y + face_h, face_x : face_x + face_w]
        roi_gray = gray[face_y : face_y + face_h, face_x : face_x + face_w]

        if detect_eyes:
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CV_8U
            )

            for (eye_x, eye_y, eye_w, eye_h) in eyes:
                if logger:
                    logger.info("Eyes detected at coordinates ({}, {})".format(eye_x, eye_y))

                cv2.rectangle(roi_frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)

        if detect_smile:
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CV_8U
            )
        
            for (smile_x, smile_y, smile_w, smile_h) in smiles:
                cv2.rectangle(roi_frame, (smile_x, smile_y), (smile_x + smile_w, smile_y + smile_h), (0, 0, 255), 2)

    return frame

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)   # To show only logs from the above logger

    logger.info("Starting Video Capture ...")

    # Capture video from default webcam
    video_cap = cv2.VideoCapture(0)

    # Iterate over an infinite loop
    while True:
        # Read a single frame
        _, frame = video_cap.read()

        # Detect faces and eyes in this frame
        result = detect_n_plot(frame, detect_smile=False, logger=logger)

        # Show the detection result
        cv2.imshow("Real-time detection", result)

        # Set the 'q' button to quit the program
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    video_cap.release()
    cv2.destroyAllWindows()