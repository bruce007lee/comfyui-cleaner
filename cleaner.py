from PIL import Image
from modules.logger import logger
from modules import devices
import platform
import os
import cv2
import numpy as np
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from torchvision import transforms

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert (
        init_image.size == mask_image.size
    ), "The sizes of the image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height * scale + 0.5)
        resize_width = int(width * scale + 0.5)
        if height != resize_height or width != resize_width:
            logger.info(
                f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})"
            )
            init_image = transforms.functional.resize(
                init_image,
                (resize_height, resize_width),
                transforms.InterpolationMode.LANCZOS,
            )
            mask_image = transforms.functional.resize(
                mask_image,
                (resize_height, resize_width),
                transforms.InterpolationMode.LANCZOS,
            )
        if resize_height != new_height or resize_width != new_width:
            logger.info(
                f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})"
            )
            init_image = transforms.functional.center_crop(
                init_image, (new_height, new_width)
            )
            mask_image = transforms.functional.center_crop(
                mask_image, (new_height, new_width)
            )

    return init_image, mask_image


class Cleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "model_type": (
                    [
                        "lama",
                        "ldm",
                        "zits",
                        "mat",
                        "fcf",
                        "manga",
                    ],
                    {"default": "lama"},
                ),
            }
        }

    RETURN_TYPES = "IMAGE"
    RETURN_NAMES = "image"
    FUNCTION = "generate"

    CATEGORY = "Cleaner"

    def generate(self, image, mask, model_type):
        global sam_dict

        if image is None or sam_dict["mask_image"] is None or mask is None:
            logger.error("The image or mask does not exist")
            return None

        mask_image = sam_dict["mask_image"]
        if image.shape != mask_image.shape:
            logger.error("The sizes of the image and mask do not match")
            return None

        # save_mask_image(mask_image, cleaner_save_mask_chk)

        logger.info(f"Loading model {model_type}")
        if platform.system() == "Darwin":
            model = ModelManager(name=model_type, device=devices.cpu)
        else:
            model = ModelManager(name=model_type, device=devices.device)

        init_image, mask_image = auto_resize_to_pil(image, mask_image)
        width, height = init_image.size

        init_image = np.array(init_image)
        mask_image = np.array(mask_image.convert("L"))

        config = Config(
            ldm_steps=20,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=512,
            hd_strategy_resize_limit=512,
            prompt="",
            sd_steps=20,
            sd_sampler=SDSampler.ddim,
        )

        output_image = model(image=init_image, mask=mask_image, config=config)
        output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)

        # save_name = (
        #     "_".join([ia_file_manager.savename_prefix, os.path.basename(model_type)])
        #     + ".png"
        # )
        # save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        # output_image.save(save_name)

        del model
        return [output_image]
