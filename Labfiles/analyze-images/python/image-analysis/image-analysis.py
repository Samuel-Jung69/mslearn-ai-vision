from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys

# import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def show_objects(image_path, objects_list):
    """Annotate detected objects and save to objects.jpg (no extra console output)."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in objects_list:
            # Bounding box may be returned as x,y,w,h
            bb = obj.bounding_box
            x = int(getattr(bb, "x", 0))
            y = int(getattr(bb, "y", 0))
            w = int(getattr(bb, "w", 0))
            h = int(getattr(bb, "h", 0))
            draw.rectangle([x, y, x + w, y + h], outline="cyan", width=3)
            # Label with first tag name (as used in the lab's printout)
            if obj.tags and len(obj.tags) > 0:
                name = obj.tags[0].name
                draw.text((x + 4, y + 4), name, fill="cyan")
        image.save("objects.jpg")
    except Exception:
        # Silent fail to avoid extra console text not specified by the lab
        pass


def show_people(image_path, people_list):
    """Annotate detected people and save to people.jpg (no extra console output)."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for person in people_list:
            bb = person.bounding_box
            x = int(getattr(bb, "x", 0))
            y = int(getattr(bb, "y", 0))
            w = int(getattr(bb, "w", 0))
            h = int(getattr(bb, "h", 0))
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=3)
        image.save("people.jpg")
    except Exception:
        # Silent fail to avoid extra console text not specified by the lab
        pass


def main():
    try:
        # Get configuration settings
        load_dotenv()
        ai_endpoint = os.getenv("AI_SERVICE_ENDPOINT")
        ai_key = os.getenv("AI_SERVICE_KEY")

        # Get image
        image_file = "images/street.jpg"
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key))

        # Analyze image
        with open(image_file, "rb") as f:
            image_data = f.read()
        print(f'\nAnalyzing {image_file}\n')

        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.TAGS,
                VisualFeatures.OBJECTS, VisualFeatures.PEOPLE],
        )

        # Get image captions
        if result.caption is not None:
            print("\nCaption:")
            print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

            if result.dense_captions is not None:
                print("\nDense Captions:")
                for caption in result.dense_captions.list:
                    print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

        # Get image tags
        if result.tags is not None:
            print("\nTags:")
            for tag in result.tags.list:
                print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))

        # Get objects in the image
        if result.objects is not None:
            print("\nObjects in image:")
            for detected_object in result.objects.list:
                # Print object tag and confidence
                print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))
            # Annotate objects in the image
            show_objects(image_file, result.objects.list)

        # Get people in the image
        if result.people is not None:
            print("\nPeople in image:")
            for detected_person in result.people.list:
                if detected_person.confidence > 0.2:
                    # Print location and confidence of each person detected
                    print(" {} (confidence: {:.2f}%)".format(detected_person.bounding_box, detected_person.confidence * 100))
            # Annotate people in the image
            show_people(image_file, result.people.list)

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
