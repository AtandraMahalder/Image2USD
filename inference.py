import time
import os
import math
from natsort import natsorted
# set environment variables 
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from peft import PeftModel
from PIL import Image
from utils.utils import create_stage, extract_prims_from_encoding_without_light
from utils.utils_classes import rectlight, domelight
import argparse
from tqdm import tqdm
import json

system_message = """You are a vision model that learns to convert images into structured descriptions. Your task is to analyze images and generate descriptions using the specific format and encoding demonstrated in your training data.

Given an image, generate a description that captures the visual content using the format, syntax, and conventions present in your training examples.

Your objectives:
- Analyze the visual content in the image
- Learn and reproduce the specific format and structure from your training dataset
- Generate descriptions that match the syntax, organization, and representational approach demonstrated in your training examples
- Maintain consistency with the learned format throughout your output

Adapt to whatever format is present in your training data:
- Syntax and structural conventions
- Vocabulary and notation systems
- Organizational patterns and hierarchy
- Representational approach and level of detail

Critical requirements:
- Output ONLY the encoded description
- Do not add explanatory text, comments, or natural language descriptions
- Do not include phrases like "Here is the encoding:" or "The description is:"
- Match the exact format, syntax, and structure from your training examples
- If uncertain about any detail, use the most likely representation based on training patterns
- Do not indicate uncertainty or add qualifiers

Generate only the description following the format learned from your training examples. Begin generating immediately without any preamble."""

def ensure_reproducibility(seed=0):
    # Set all random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

def format_data(image):
    return [{
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    }
                ]
           }]

def generate_text_from_sample(model, processor, images, logits_processor=None, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    samples = [format_data(image) for image in images]
    text_inputs = [processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True) for sample in samples]

    # Process the visual input from the sample
    image_inputs = [process_vision_info(sample) for sample in samples]

    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_inputs,
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device
    
    # Generate text with the model
    if logits_processor:
        init_time = time.time()
        generated_ids = model.generate(**model_inputs, logits_processor=[logits_processor], max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()
    else:
        init_time = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text, len(trimmed_generated_ids[0]) - 1, end_time - init_time

def get_lighting(prims, light_params):
    r, g, b, I = light_params
    r = r.item()
    g = g.item()
    b = b.item()
    I = 10**(I.item())
    for prim in prims:
        if prim.type == 'warehouse_1':
            return domelight([r, g, b], I)
        if prim.type == 'warehouse_2':
            return rectlight([r, g, b], I, [48.709, 404, 587.178], 1500, 2500)
        if prim.type == 'room':
            return rectlight([r, g, b], I, [0, -125.829, 328.544], 500, 50)
        
parser = argparse.ArgumentParser()
parser.add_argument('--eval_dataset', default='test_data')
parser.add_argument('--vlm_model_dir', default='finetuned_gemma')
parser.add_argument('--cnn_model_dir', default='finetuned_resnet/checkpoint-20')
parser.add_argument('--percent_eval', type=float, default=1)
parser.add_argument('--continue_generation', action="store_true")
parser.add_argument('--encoding_scheme', choices=['p10', 'p1000', 'b1999', 'fp15'], default='fp15')
parser.add_argument('--asset_dir', default='assets')
parser.add_argument('--suffix', default='image_2_usd')
parser.add_argument('--batch_size', type=int, default=30)
args = parser.parse_args()

asset_path = os.path.realpath(args.asset_dir)
vlm_model = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E4B-it", dtype=torch.bfloat16, device_map='auto')
processor = AutoProcessor.from_pretrained(args.vlm_model_dir, use_fast=True)
vlm_model.resize_token_embeddings(len(processor.tokenizer))
vlm_model = PeftModel.from_pretrained(vlm_model, args.vlm_model_dir)
vlm_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cnn_model = models.resnet50()
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
state_dict = torch.load(args.cnn_model_dir, weights_only=True)
cnn_model.load_state_dict(state_dict)
cnn_model.eval()

torch.set_grad_enabled(False)

with open(f"{args.encoding_scheme}_grammar.ebnf", "r") as file:
    grammar_str = file.read()
grammar = IncrementalGrammarConstraint(grammar_str, "root", processor.tokenizer)

max_objects = natsorted(os.listdir(args.eval_dataset))
image_batch = []
address_batch = []
for num_obj in tqdm(max_objects, desc='Object Count'):
    samples = natsorted(os.listdir(os.path.join(args.eval_dataset, num_obj)))

    num_to_eval = math.floor(args.percent_eval * len(samples))
    samples = samples[:num_to_eval]

    for sample in tqdm(samples, desc='Samples'):
        if args.continue_generation:
            if os.path.exists(os.path.join(args.eval_dataset, num_obj, sample, f'generation_stats_{args.suffix}.txt')):
                continue
                
        image_batch.append(Image.open(os.path.join(args.eval_dataset, num_obj, sample, 'image.png')))
        address_batch.append(os.path.join(args.eval_dataset, num_obj, sample))
        if len(image_batch) == args.batch_size:
            ensure_reproducibility(0)
            encoded_output, num_tokens, generation_time = generate_text_from_sample(vlm_model, processor, image_batch, None, 5000)
            image_tensor = torch.stack([preprocess(img.convert('RGB')) for img in image_batch])
            image_tensor.to('cuda')
            light_params = cnn_model(image_tensor)
            for output, light_param, address in zip(encoded_output, list(light_params), address_batch):
                try:
                    entities = extract_prims_from_encoding_without_light(output, args.encoding_scheme, asset_path)
                    if len(entities['cameras']) == 1:
                        light = get_lighting(entities['prims'], light_param)
                        if light is None:
                            # no background prim detected -> couldn't determine lighting;
                            # record the output and move on (no scene written, no GCD retry)
                            with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                                f.write(f'Encoded Output: {output}\n')
                                f.write(f'Number of Tokens: {num_tokens}\n')
                                f.write(f'Generation Time: {generation_time}\n')
                            continue
                        test_stage_path = os.path.join(address, f'pred_{args.suffix}.usda')
                        create_stage(test_stage_path, entities['prims'], light)
                        
                        test_stage_cam = os.path.join(address, f'pred_camera_{args.suffix}.json')
                        camera_loc = entities['cameras'][0].location
                        camera_look_at = entities['cameras'][0].look_at
                        camera_data = {
                            'location': [float(val) for val in list(camera_loc)],
                            'look_at': [float(val) for val in list(camera_look_at)]
                        }
                        
                        with open(test_stage_cam, 'w') as f:
                            json.dump(camera_data, f)
                    else:
                        raise Exception('Invalid Encoding')
                        
                    with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                        f.write(f'Encoded Output: {output}\n')
                        f.write(f'Number of Tokens: {num_tokens}\n')
                        f.write(f'Generation Time: {generation_time}\n')
                
                except Exception as e:
                    ensure_reproducibility(0)
                    grammar.reset()
                    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
                    img = [Image.open(os.path.join(address, 'image.png'))]
                    gcd_output, gcd_tokens, gcd_generation_time = generate_text_from_sample(vlm_model, processor, img, grammar_processor, 5000)
                    entities = extract_prims_from_encoding_without_light(gcd_output[0], args.encoding_scheme, asset_path)
                    light = get_lighting(entities['prims'], light_param)
                    if light is None:
                        # no background prim detected -> couldn't determine lighting; record output and move on
                        with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                            f.write(f'Encoded Output: {gcd_output}\n')
                            f.write(f'Number of Tokens: {gcd_tokens}\n')
                            f.write(f'Generation Time: {gcd_generation_time}\n')
                        continue
                    test_stage_path = os.path.join(address, f'pred_{args.suffix}.usda')
                    create_stage(test_stage_path, entities['prims'], light)
                    
                    test_stage_cam = os.path.join(address, f'pred_camera_{args.suffix}.json')
                    camera_loc = entities['cameras'][0].location
                    camera_look_at = entities['cameras'][0].look_at
                    camera_data = {
                        'location': [float(val) for val in list(camera_loc)],
                        'look_at': [float(val) for val in list(camera_look_at)]
                    }
                    
                    with open(test_stage_cam, 'w') as f:
                        json.dump(camera_data, f)
                    with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                        f.write(f'Encoded Output: {gcd_output}\n')
                        f.write(f'Number of Tokens: {gcd_tokens}\n')
                        f.write(f'Generation Time: {gcd_generation_time}\n')
            
            for img in image_batch:
                img.close()
            image_batch = []
            address_batch = []
    
    if len(image_batch) > 0:
        ensure_reproducibility(0)
        encoded_output, num_tokens, generation_time = generate_text_from_sample(vlm_model, processor, image_batch, None, 5000)
        image_tensor = torch.stack([preprocess(img.convert('RGB')) for img in image_batch])
        image_tensor.to('cuda')
        light_params = cnn_model(image_tensor)
        for output, light_param, address in zip(encoded_output, list(light_params), address_batch):
            try:
                entities = extract_prims_from_encoding_without_light(output, args.encoding_scheme, asset_path)
                if len(entities['cameras']) == 1:
                    light = get_lighting(entities['prims'], light_param)
                    if light is None:
                        # no background prim detected -> couldn't determine lighting;
                        # record the output and move on (no scene written, no GCD retry)
                        with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                            f.write(f'Encoded Output: {output}\n')
                            f.write(f'Number of Tokens: {num_tokens}\n')
                            f.write(f'Generation Time: {generation_time}\n')
                        continue
                    test_stage_path = os.path.join(address, f'pred_{args.suffix}.usda')
                    create_stage(test_stage_path, entities['prims'], light)
                        
                    test_stage_cam = os.path.join(address, f'pred_camera_{args.suffix}.json')
                    camera_loc = entities['cameras'][0].location
                    camera_look_at = entities['cameras'][0].look_at
                    camera_data = {
                        'location': [float(val) for val in list(camera_loc)],
                        'look_at': [float(val) for val in list(camera_look_at)]
                    }
                        
                    with open(test_stage_cam, 'w') as f:
                        json.dump(camera_data, f)
                else:
                    raise Exception('Invalid Encoding')
                
                with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                    f.write(f'Encoded Output: {output}\n')
                    f.write(f'Number of Tokens: {num_tokens}\n')
                    f.write(f'Generation Time: {generation_time}\n')
            
            except Exception as e:
                ensure_reproducibility(0)
                grammar.reset()
                grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
                img = [Image.open(os.path.join(address, 'image.png'))]
                gcd_output, gcd_tokens, gcd_generation_time = generate_text_from_sample(vlm_model, processor, img, grammar_processor, 5000)
                entities = extract_prims_from_encoding_without_light(gcd_output[0], args.encoding_scheme, asset_path)
                light = get_lighting(entities['prims'], light_param)
                if light is None:
                    # no background prim detected -> couldn't determine lighting; record output and move on
                    with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                        f.write(f'Encoded Output: {gcd_output}\n')
                        f.write(f'Number of Tokens: {gcd_tokens}\n')
                        f.write(f'Generation Time: {gcd_generation_time}\n')
                    continue
                test_stage_path = os.path.join(address, f'pred_{args.suffix}.usda')
                create_stage(test_stage_path, entities['prims'], light)
                    
                test_stage_cam = os.path.join(address, f'pred_camera_{args.suffix}.json')
                camera_loc = entities['cameras'][0].location
                camera_look_at = entities['cameras'][0].look_at
                camera_data = {
                    'location': [float(val) for val in list(camera_loc)],
                    'look_at': [float(val) for val in list(camera_look_at)]
                }

                with open(test_stage_cam, 'w') as f:
                    json.dump(camera_data, f)
                with open(os.path.join(address, f'generation_stats_{args.suffix}.txt'), 'w') as f:
                    f.write(f'Encoded Output: {gcd_output}\n')
                    f.write(f'Number of Tokens: {gcd_tokens}\n')
                    f.write(f'Generation Time: {gcd_generation_time}\n')
        
        for img in image_batch:
            img.close()
        image_batch = []
        address_batch = []