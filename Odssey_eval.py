import json
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
from src.model_file.LLM_compression_v2_action.modeling_qwen2vl import Qwen2VLForConditionalGeneration
import argparse

IGNORE_INDEX = -100
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SYSTEM_MESSAGE = "You are a helpful assistant."

# Use the same prompt format as in GUIOdyssey_process.py
PROMPT_ORIGIN = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--model_path', type=str, default="/data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/Models/guiodssey_bs64_lr3e-5_mask535_drop6_alpha1_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=4,
                    help='History length for evaluation')
parser.add_argument('--drop_k', type=int, default=6,
                    help='Drop k for model')
parser.add_argument('--test_file', type=str, 
                    default="/data7/Users/zxr/hyh/SimpAgent-main/data/guiodssey/guiodyssey_test_llavaformat.json",
                    help='Path to test dataset file')
parser.add_argument('--alpha', type=int, default=1,
                    help='The path where the model is loaded from')

# 解析参数
args = parser.parse_args()

# GUI Odyssey data directories (same as GUIOdssey_process.py)
DATA_DIR = "/data1/GUIData/GUI-Odyssey/data"
pic_base = os.path.join(DATA_DIR, 'screenshots')

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
)
min_pixels = 200704
max_pixels = 1003520
processor = AutoProcessor.from_pretrained("/data1/Models/Qwen2-VL-2B-Instruct")
model.drop_k = args.drop_k
model.model.drop_k = args.drop_k
model.alpha = args.alpha

def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=1280 * 28 * 28):
    """Process image for model input"""
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixels": min_pixel,
                "max_pixels": max_pixel,
            }
            ]
        }
    ]

    image_input, _ = process_vision_info_with_resize(messages)
    return image_input[0]

def generate_grounding(image_paths, query):
    """Generate action prediction from model"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        }
    ]

    images = []
    for image_file in image_paths:
        images.append(get_image_info(image_file))

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text = text.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    return output_text

def main():
    """Main evaluation function for GUI Odyssey dataset"""
    print(f"Starting GUI Odyssey evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"History length: {args.his_num}")
    
    # Load test dataset
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    
    with open(args.test_file, 'r') as f:
        all_test_data = json.load(f)

    test_data = all_test_data
    
    print(f"Loaded {len(test_data)} test samples")
    
    outputs = []
    img_not_found = 0
    
    # Process each test sample
    for sample_idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            conversations = sample.get("conversations", [])
            image_paths = sample.get("image", [])
            
            if len(conversations) < 2 or len(image_paths) < 1:
                print(f"Warning: Sample {sample_idx} has insufficient data")
                continue
            
            # Extract user prompt and ground truth action
            # In the format from GUIOdssey_process.py, conversations[0] is human, conversations[1] is assistant
            user_content = conversations[0].get("value", "")
            gt_action = conversations[1].get("value", "")
            
            # Check if all images exist
            missing_images = []
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    missing_images.append(img_path)
                    img_not_found += 1
            
            if missing_images:
                print(f"Warning: Sample {sample_idx} has {len(missing_images)} missing images")
                continue
            
            # Get model prediction
            pred_action = generate_grounding(image_paths, user_content)
            
            # Store result
            outputs.append({
                'sample_id': sample_idx,
                'pred': pred_action,
                'gt': gt_action,
            })
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Print statistics
    print(f"\nEvaluation completed!")
    print(f"Total samples processed: {len(outputs)}")
    print(f"Images not found: {img_not_found}")
    
    # Save results
    savefile = os.path.join("/data7/Users/zxr/hyh/SimpAgent-main/odssey_eval/guiodssey_bs64_lr3e-5_mask535_drop6_alpha1_eval","odssey_results.json")
    json.dump(outputs, open(savefile, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f"Results saved to: {savefile}")

if __name__ == "__main__":
    main()



