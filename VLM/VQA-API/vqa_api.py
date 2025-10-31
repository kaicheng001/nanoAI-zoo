import logging
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io
import os
from typing import Optional

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def encode_image(image_array: np.ndarray) -> str:
    """
    Encode a numpy HWC image (RGB or RGBA) to base64 PNG string.
    """
    try:
        pil_image = Image.fromarray(image_array[:, :, :3], mode="RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")


class BaseVQAModel:
    """
    Unifies a simple VQA interface: answer(image, question) -> str.
    Implementations should internally call their existing chat APIs.
    """

    def answer(self, image_array: np.ndarray, question: str) -> str:
        raise NotImplementedError

    def reset(self):
        pass

    def get_spend(self) -> float:
        raise NotImplementedError


class GeminiVLM(BaseVQAModel):
    """
    Gemini-backed VQA wrapper that preserves the original call logic.
    """

    def __init__(
        self, model: str = "gemini-2.5-flash", system_instruction: Optional[str] = None
    ):
        self.name = model
        self.client = OpenAI(
            api_key="your_api_key",
            base_url="base_url",
        )
        self.system_instruction = system_instruction
        self.spend = 0.0
        if "1.5-flash" in self.name:
            self.cost_per_input_token = 0.075 / 1_000_000
            self.cost_per_output_token = 0.3 / 1_000_000
        elif "1.5-pro" in self.name:
            self.cost_per_input_token = 1.25 / 1_000_000
            self.cost_per_output_token = 5 / 1_000_000
        else:
            self.cost_per_input_token = 0.1 / 1_000_000
            self.cost_per_output_token = 0.4 / 1_000_000

    def _call_chat(self, image_array: np.ndarray, text_prompt: str) -> str:
        base64_image = encode_image(image_array)
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    },
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False,
            )
            self.spend += (
                response.usage.prompt_tokens * self.cost_per_input_token
                + response.usage.completion_tokens * self.cost_per_output_token
            )
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def _call(self, image_array: np.ndarray, text_prompt: str) -> str:
        base64_image = encode_image(image_array)
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False,
            )
            self.spend += (
                response.usage.prompt_tokens * self.cost_per_input_token
                + response.usage.completion_tokens * self.cost_per_output_token
            )
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def answer(self, image_array: np.ndarray, question: str) -> str:
        """
        VQA entry point: given an image and a question, return an answer.
        Preserves the original behavior: if a system instruction exists, use chat-flow; otherwise use user-only flow.
        """
        if self.system_instruction:
            return self._call_chat(image_array, question)
        return self._call(image_array, question)

    def get_spend(self) -> float:
        return float(self.spend)


class QwenVLM(BaseVQAModel):
    """
    Qwen-backed VQA wrapper that preserves the original call logic.
    """

    def __init__(
        self,
        model: str = "qwen-max",
        system_instruction: Optional[str] = None,
    ):
        self.name = model
        self.client = OpenAI(
            api_key="your_api_key",
            base_url="base_url",
        )
        self.system_instruction = system_instruction
        self.spend = 0.0

    def _call_chat(self, image_array: np.ndarray, text_prompt: str) -> str:
        base64_image = encode_image(image_array)
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False,
            )
            self.spend += (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def _call(self, image_array: np.ndarray, text_prompt: str) -> str:
        base64_image = encode_image(image_array)
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                stream=False,
            )
            self.spend += (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
        except Exception as e:
            print(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"
        return response.choices[0].message.content

    def answer(self, image_array: np.ndarray, question: str) -> str:
        """
        VQA entry point: given an image and a question, return an answer.
        Preserves the original behavior: if a system instruction exists, use chat-flow; otherwise use user-only flow.
        """
        if self.system_instruction:
            return self._call_chat(image_array, question)
        return self._call(image_array, question)

    def get_spend(self) -> float:
        return float(self.spend)


if __name__ == "__main__":
    import sys
    import pathlib
    from PIL import Image

    # ensure project root is on sys.path
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from utils.load_data import load_single_image

    # Example usage:
    img = load_single_image("assets/img.jpg", image_size=224).squeeze(0).numpy()
    q = "What is shown in the image?"
    vqa = GeminiVLM(system_instruction="You are a helpful VQA assistant.")
    print(vqa.answer(img, q))
    vqa_qwen = QwenVLM(system_instruction="You are a helpful VQA assistant.")
    print(vqa_qwen.answer(img, q))
