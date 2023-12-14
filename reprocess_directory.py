import math
import re

import sys
import traceback

import os
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr

from modules import images as imgutil
from modules.generation_parameters_copypaste import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts as scripts
from modules.processing import Processed, process_images

class Script(scripts.Script):
    def title(self):
        return "Get prompt from batch of images"

    def ui(self, is_img2img):
        input_dir = gr.Textbox(label="input_dir")
        return [input_dir]

    def run(self, p, input_dir):
        start_images = shared.listfiles(input_dir)

        all_images = []
        batch_count = p.n_iter
        state.job_count = len(start_images) * p.n_iter
        initial_seed = None
        initial_info = None
        
        # extract "default" params to use in case getting png info fails
        prompt = p.prompt
        negative_prompt = p.negative_prompt
        seed = p.seed
        cfg_scale = p.cfg_scale
        sampler_name = p.sampler_name
        steps = p.steps
        
        for i, image in enumerate(start_images):
            
            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            try:
                img = Image.open(image)
            except UnidentifiedImageError as e:
                print(e)
                continue
                
            try:
                info_img = img
                geninfo, _ = imgutil.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
            except Exception as e:
                print(e)
                parsed_parameters = {}

            result = re.search('Seed: (.*), Size', geninfo)
            parsed_seed = result.group(1)

            p.prompt = parsed_parameters["Prompt"]
            p.seed = int(parsed_seed)

            proc = process_images(p)
                    
            last_image = proc.images[0]
            initial_info = proc.info
            initial_seed = p.seed
            all_images.append(last_image)

        return Processed(p, all_images, initial_seed, initial_info)
