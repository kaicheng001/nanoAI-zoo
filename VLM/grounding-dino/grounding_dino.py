from typing import Optional
import sys
import numpy as np
import torch

sys.path.insert(0, "GroundingDINO/")
import torchvision.transforms.functional as F
import groundingdino.datasets.transforms as T
from detections import ObjectDetections
from server_wrapper import ServerMixin, host_model, send_request, str_to_image


try:
    from groundingdino.util.inference import load_model, predict
except ModuleNotFoundError:
    print(
        "Could not import groundingdino. This is OK if you are only using the client."
    )
sys.path.pop(0)

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . sheep ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.caption = caption

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
    ) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("GroundingDINO is detecting. Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # Remove detections whose class names do not exactly match the provided classes
        # classes = caption_to_use[: -len(" .")].split(" . ")
        # detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(
        self,
        image_numpy: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                caption=payload["caption"],
                box_threshold=payload["box_threshold"],
                text_threshold=payload["text_threshold"],
            ).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
