from dotenv import load_dotenv
import os
import sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential

def annotate_faces(image_file, detected_faces):
    """
    Draw bounding boxes for each detected face and save to 'detected_faces.jpg'.
    Uses the face_rectangle (left, top, width, height) from each detected face.
    """
    # Open image
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = 'cyan'
    width = 3

    # Draw rectangle for each face
    for face in detected_faces:
        r = face.face_rectangle
        # rectangle expects [left, top, right, bottom]
        bbox = [(r.left, r.top), (r.left + r.width, r.top + r.height)]
        draw.rectangle(bbox, outline=color, width=width)

    # Save and show
    out_file = 'detected_faces.jpg'
    image.save(out_file)
    # If running with display available, show the result
    try:
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.imshow(image)
        plt.show()
    except Exception:
        pass
    print(f"Annotated image saved to {out_file}")

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        cog_key = os.getenv('COG_SERVICE_KEY')

        if not cog_endpoint or not cog_key:
            raise ValueError("COG_SERVICE_ENDPOINT / COG_SERVICE_KEY not set in .env")

        # Determine image to analyze
        image_file = 'images/face1.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")

        # Authenticate Face client
        face_client = FaceClient(
            endpoint=cog_endpoint,
            credential=AzureKeyCredential(cog_key))

        # Specify facial features to be retrieved
        features = [FaceAttributeTypeDetection01.HEAD_POSE, FaceAttributeTypeDetection01.OCCLUSION, FaceAttributeTypeDetection01.ACCESSORIES]

        # Get faces
        with open(image_file, mode="rb") as image_data:
            detected_faces = face_client.detect(
                image_content=image_data.read(),
                detection_model=FaceDetectionModel.DETECTION01,
                recognition_model=FaceRecognitionModel.RECOGNITION01,
                return_face_id=False,
                return_face_attributes=features,
            )

        face_count = 0
        if len(detected_faces) > 0:
            print(len(detected_faces), 'faces detected.')
            for face in detected_faces:
                # Get face properties
                face_count += 1
                print('\nFace number {}'.format(face_count))
                print(' - Head Pose (Yaw): {}'.format(face.face_attributes.head_pose.yaw))
                print(' - Head Pose (Pitch): {}'.format(face.face_attributes.head_pose.pitch))
                print(' - Head Pose (Roll): {}'.format(face.face_attributes.head_pose.roll))
                print(' - Forehead occluded?: {}'.format(face.face_attributes.occlusion["foreheadOccluded"]))
                print(' - Eye occluded?: {}'.format(face.face_attributes.occlusion["eyeOccluded"]))
                print(' - Mouth occluded?: {}'.format(face.face_attributes.occlusion["mouthOccluded"]))
                print(' - Accessories:')
                for accessory in face.face_attributes.accessories:
                    print('   - {}'.format(accessory.type))

            # Annotate faces in the image
            annotate_faces(image_file, detected_faces)
        else:
            print("No faces detected.")

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
