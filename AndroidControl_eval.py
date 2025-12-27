import json
import os
import re
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
from PIL import Image, ImageDraw
from src.training.my_qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from src.model_file.LLM_compression_v2_action.modeling_qwen2vl import Qwen2VLForConditionalGeneration
import pdb
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/data7/Users/zxr/hyh/SimpAgent-main/androidcontrol_eval/androidcontrol_bs128_lr3e-4_mask535_drop6_eval",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/Models/androidcontrol_bs128_lr3e-4_mask535_drop6_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=2,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=6,
                    help='The path where the model is loaded from')
parser.add_argument('--alpha', type=int, default=1,
                    help='The path where the model is loaded from')



# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + '.json'

import torch
# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
    args.model_path , torch_dtype=torch.bfloat16, device_map="cuda"
    #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
min_pixels = 200704
max_pixels = 1003520
processor = AutoProcessor.from_pretrained("/data1/Models/Qwen2-VL-2B-Instruct")
model.drop_k = args.drop_k
model.model.drop_k = args.drop_k
model.alpha = args.alpha


def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=1280 * 28 * 28):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
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

    image_input, _ = process_vision_info(messages)

    return image_input[0]


def generate_grounding(image_path, query):
    # TODO
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": query},
                #  "text": f"{query}\n请告诉我怎么操作同时输出坐标。"},
            ],
        }
    ]

    images = []
    for image_file in image_path:
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
def android_action2step(action_str, img_width, img_height):
    """
    将AndroidControl的动作字符串转换为标准格式
    支持所有8种动作类型: click, scroll, input_text, wait, open_app, navigate_back, long_press, navigate_home
    使用图片的实际尺寸进行坐标归一化
    """
    try:
        action_data = json.loads(action_str)
        action_type = action_data["action_type"]
        
        if action_type == "click" or action_type == "long_press":
            if action_type == "click":
               action_type_id = 4
            else:
                action_type_id = 11
                
            x = action_data["x"]
            y = action_data["y"]
            # 使用图片实际尺寸进行归一化到0-1000范围
            if img_width > 0 and img_height > 0:
                x_norm = int(1000 * x / img_width)
                y_norm = int(1000 * y / img_height)
                # 确保坐标在有效范围内
                x_norm = max(0, min(1000, x_norm))
                y_norm = max(0, min(1000, y_norm))
            else:
                # 如果尺寸无效，使用默认归一化
                x_norm = int(1000 * x / 2000)
                y_norm = int(1000 * y / 2000)
                
            return f'{{"action_type": {action_type_id}, "click_point": ({x_norm},{y_norm})}}'
        
        elif action_type == "input_text":
            action_type_id = 3
            text = action_data["text"]
            return f'{{"action_type": {action_type_id}, "typed_text": "{text}"}}'
        
        elif action_type == "scroll":
            direction = action_data["direction"]
            if direction == "down":
                action_type_id = 0
            elif direction == "up":
                action_type_id = 1
            elif direction == "left":
                action_type_id = 8
            elif direction == "right":
                action_type_id = 9
            else:
                action_type_id = 0
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "wait":
            action_type_id = 2
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "navigate_back":
            action_type_id = 5
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "open_app":
            action_type_id = 7
            app_name = action_data["app_name"]
            return f'{{"action_type": {action_type_id}, "app_name": "{app_name}"}}'
        
        elif action_type == "navigate_home":
            action_type_id = 6
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "finish":
            action_type_id = 10
            return f'{{"action_type": {action_type_id}}}'
        
        else:
            print(f"未知动作类型: {action_type}")
            return f'{{"action_type": 99}}'
            
    except Exception as e:
        print(f"Error parsing action: {action_str}, error: {e}")
        return f'{{"action_type": 99}}'

import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
import numpy as np

import json
import os
from PIL import Image
from tqdm import tqdm
prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}\n"
with open('/data7/Users/zxr/zhouxurui/GUI-dataset/AndroidControl/splits.json', 'r') as file:
    splits=json.load(file)
    test_ids=splits['test']
    test_ids = list(set(test_ids))
    # test_ids = test_ids[0:len(test_ids) // 4]
outputs = []
with open('/data7/Users/zxr/hyh/SimpAgent-main/data/AndroidControl/androidcontrol_data_test.json', 'r') as file:
    data = json.load(file)


img_not_found = 0
llava_format_data = []
img_dir = "/data1/GUIData/AndroidControl/androidcontrol_images/"
for episode in tqdm(data):
    screenshot_widths = episode.get("screenshot_widths", [])
    screenshot_heights=episode.get("screenshot_heights", [])
    previous_imgs = []
    previous_actions = []
    flag = 0
    if episode['episode_id'] not in test_ids:
        # print(episode['episode_id'])
        continue
    for idx in range(len(episode['actions'])):
        step_data = {}

        img_filename = episode["images"][idx]
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            print('image not found')
            flag = 1
            continue
        img_width = screenshot_widths[idx] if idx < len(screenshot_widths) else 0
        img_height = screenshot_heights[idx] if idx < len(screenshot_heights) else 0


        goal = episode["goal"]

        prompt = prompt_origin.format(goal)

        cur_step_preimg = previous_imgs[-2:]
        cur_step_idx = len(previous_imgs[-2:])
        cur_all_imgs = []
        for i, action in enumerate(previous_actions[-2:]):
            prompt += 'Image_' + str(i) + ":<image>\n\n"
            prompt += 'Step_' + str(i) + ':' + action + " .\n"
            cur_all_imgs.append(previous_imgs[-2:][i])

        action_str = episode['actions'][idx]
        action_step = android_action2step(action_str, img_width, img_height)

        previous_actions.append(action_step)
        previous_imgs.append(img_path)

        conversations = []
        prompt += 'Image_' + str(cur_step_idx) + ":<image>\n\n"
        # print(cur_all_imgs)
        cur_all_imgs.append(img_path)

        response = generate_grounding(cur_all_imgs, prompt)
        outputs.append({
                    'episode_id' : episode['episode_id'],
                    'pred': response,
                    'gt': action_step,
                })
        #print(outputs[-1])



print(f"Saving predict result ...")
# time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
savefile = os.path.join("/data7/Users/zxr/hyh/SimpAgent-main/androidcontrol_eval/androidcontrol_bs128_lr3e-4_mask535_drop6_eval","AndroidControl_results.json")
json.dump(outputs, open(savefile, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)



