import numpy as np
import logging
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

class VisualPrompter:
    def __init__(self, model_path: str = "./playground/Pretrained_models/qwen3_vl_2B"):
        """
        Initialize the VisualPrompter with a local Qwen3-VL 2B model.
        """
        logging.info(f"VisualPrompter loading local Qwen3-VL model from {model_path}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logging.info("VisualPrompter initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to load local Qwen3-VL model from {model_path}. Error: {e}")
            self.model = None
            self.processor = None

    def add_visual_prompt(self, img: np.ndarray, wrist_img: np.ndarray, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Modify the images by querying the local Qwen3-VL model for bounding boxes
        related to the instruction, and draw them on the primary image.

        Args:
            img: The primary camera image (H, W, C), uint8.
            wrist_img: The wrist camera image (H, W, C), uint8.
            instruction: The language instruction for the current task.

        Returns:
            A tuple of (modified_img, modified_wrist_img).
        """
        if self.model is None or self.processor is None:
            return img, wrist_img

        # We primarily run object detection on the primary image `img`.
        # Qwen-VL accepts PIL images or local file paths.
        pil_img = Image.fromarray(img)

        # Construct prompt requesting bounding boxes
        prompt = f"Find the objects related to this instruction: '{instruction}'. Output their bounding boxes."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate output
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            logging.info(f"Qwen-VL Output for instruction '{instruction}': {output_text}")

            # Simple naive parse for bounding boxes, e.g., assuming [ymin, xmin, ymax, xmax] format in 0-1000 scale
            # Note: The exact parsing regex might need adjustment based on the specific prompt's returned format.
            import re
            # look for something like <box> (ymin, xmin), (ymax, xmax) </box> or simply [ymin, xmin, ymax, xmax]
            bboxes = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', output_text)

            modified_img = img.copy()
            H, W, _ = modified_img.shape

            for bbox in bboxes:
                ymin, xmin, ymax, xmax = map(int, bbox)
                # Qwen-VL bounding boxes are usually normalized to 1000
                ymin = int((ymin / 1000) * H)
                xmin = int((xmin / 1000) * W)
                ymax = int((ymax / 1000) * H)
                xmax = int((xmax / 1000) * W)

                # Draw green bounding box
                cv2.rectangle(modified_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            return modified_img, wrist_img

        except Exception as e:
            logging.error(f"Inference error in VisualPrompter: {e}")
            return img, wrist_img
