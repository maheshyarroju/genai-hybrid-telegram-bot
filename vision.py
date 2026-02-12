from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class ImageDescriber:
    def __init__(self):
        # Load the BLIP processor
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # Load the BLIP model
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # Set model to eval mode (best practice for inference)
        self.model.eval()

    def describe(self, image_path: str):
        """
        Takes an image path and returns:
        1) Caption
        2) Top 3 tags extracted from caption
        """

        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Convert image into model input tensors
        inputs = self.processor(image, return_tensors="pt")

        # Generate caption without gradients (faster + less memory)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)

        # Decode output tokens into text caption
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        # Extract tags from caption
        words = [w.strip(".,!?").lower() for w in caption.split()]
        tags = []

        for w in words:
            if w not in tags and len(w) > 3:
                tags.append(w)
            if len(tags) == 3:
                break

        # IMPORTANT: return both caption and tags
        return caption, tags
