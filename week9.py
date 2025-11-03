import cv2
import matplotlib.pyplot as plt

# Load pre-trained Haar Cascade classifier for face detection
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def detect_faces(image_path):
    """Detect faces in an image using Haar Cascade."""
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert the image to RGB for displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

    print(f"Detected {len(faces)} face(s).")

# Test the program with an example image
image_path = "/content/wk9.jpg"  # Replace with the path to your image
detect_faces(image_path)
