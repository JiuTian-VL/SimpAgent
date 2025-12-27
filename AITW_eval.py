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
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
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
parser.add_argument('--save_path', type=str, default="eval_results/",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data6/GUIModels/model/aitw_distill+mask557",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=4,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=3,
                    help='The path where the model is loaded from')


# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + args.model_path.split('/')[-1] + "_drop_" + str(args.drop_k) + '.json'
from transformers import AutoConfig# Default: Load the model on the available device(s)
# config = AutoConfig.from_pretrained(args.model_path)
# config.drop_k = args.drop_k
import torch
device='cuda:1'
# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
    args.model_path , torch_dtype="auto", device_map=device,attn_implementation="flash_attention_2"
    #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
model.model.drop_k = args.drop_k
model.drop_k = args.drop_k
print("drop_k:", model.drop_k)
# processor = AutoProcessor.from_pretrained("OS-Copilot/OS-Atlas-Base-7B")
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v2/checkpoint-1056/")
min_pixels = 200704
max_pixels = 1003520
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v5", min_pixels=min_pixels, max_pixels=max_pixels)
# processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/OS-Atlas-Base-7B/")
# processor = AutoProcessor.from_pretrained("/nas_sh/wentao/Qwen2-VL-7B-Instruct/")
processor = AutoProcessor.from_pretrained("/data1/Models/Qwen2-VL-2B-Instruct")


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

    image_input, _ = process_vision_info_with_resize(messages)

    return image_input[0]


def generate_grounding(image_path, query):

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
    # print(text)
    # print(images)

    # print(messages)
    # print(image_inputs)
#    resized_w, resized_h = image_inputs[0].size
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
#    import pdb
#    pdb.set_trace()

    # tmp = processor.decode(inputs['input_ids'][0])
    #print(tmp)

    # print(inputs)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    return output_text

    # print(output_text)

#     image = Image.open(image_path)
#     image_w, image_h = image.size

#     # 创建绘制对象
#     draw = ImageDraw.Draw(image)

#     try:
#         coordinates = re.findall(r"\((\d+),(\d+)\)", output_text[0])

#         # 将坐标转换为整数并保存为元组
#         points = [(int(int(x) / 1000 * image_w), int(int(y) / 1000 * image_h)) for x, y in coordinates]

#         # 绘制矩形框
#         draw.rectangle(points, outline="red", width=3)
#         draw.rectangle([tuple(truth[0]), tuple(truth[1])], outline="blue", width=3)

#         save_path = "/home/wentao/project/gui_ads/test_output/" + query + str(coordinates) + ".jpg"
#         # 显示图像
#         image = image.convert("RGB")
#         image.save(save_path)
#     except:
#         print("!!!!!!!!!!!!")

# _count = 0

from tqdm import tqdm
import action_matching 

def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [int(1000 * (touch_point[0] + lift_point[0]) / 2), int(1000* (touch_point[1] + lift_point[1]) / 2)]
            click_point = [item for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action
def process_string(s):
    # 使用正则表达式匹配所有坐标点
    pattern = r'\((\d+),(\d+)\)'
    
    # 替换所有匹配的坐标点
    def replace(match):
        # 将匹配到的数字除以1000，并四舍五入到两位小数
        x = round(float(match.group(1)) / 1000, 2)
        y = round(float(match.group(2)) / 1000, 2)
        return f"({x:.2f},{y:.2f})"
    
    return re.sub(pattern, replace, s)

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

aitw_imgs_dir = "/data7/Users/zxr/zhouxurui/GUI-dataset/aitw_seeclick/aitw_images"
aitw_test = json.load(open('/data7/Users/zxr/zhouxurui/GUI-dataset/aitw_seeclick/aitw_data_test.json', 'r'))
# aitw_test = json.load(open('/home/wentao/GUIAgent-Data/aitw_seeclick/aitw_data_train.json', 'r'))

prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
score_average = 0
all_save_results = []
all_eval_results = []

for task, episodes in aitw_test.items():
    print("Task: " + task)

    corr_action = 0
    corr_type = 0
    num_text = 0
    corr_text = 0
    num_scroll = 0
    corr_scroll = 0
    num_click = 0
    corr_click = 0
    num_both_click = 0
    corr_both_click = 0
    num_wrong_format = 0
    num = 0
    
    # episodes = episodes[0:20]

    print("sample num:", len(episodes))

    for j, episode in tqdm(enumerate(episodes)):

        previous_actions = []
        previous_imgs = []

        for step in episode:
            step_json = {'task': task, 'episode': step['ep_id'], 'correct': 'no'}

            img_filename = step["img_filename"] + '.png'
            img_path = os.path.join(aitw_imgs_dir, img_filename)
            if not os.path.exists(img_path):
                print('image not found')
                continue
            if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
                continue
            image = Image.open(img_path)

            goal = step["goal"]

            prompt = prompt_origin.format(goal)

            cur_step_idx = len(previous_imgs[-args.his_num:])
            cur_all_imgs = []
            for i, action in enumerate(previous_actions[-args.his_num:]):
                prompt += 'Image_' + str(i) + ":<image>\n"
                prompt += 'Step_' + str(i) + ': ' + action + " .\n"
                cur_all_imgs.append(previous_imgs[-args.his_num:][i])

            prompt += 'Image_' + str(cur_step_idx) + ":<image>\n"
            cur_all_imgs.append(img_path)
            action_step = action2step(step)

            previous_actions.append(action_step)
            previous_imgs.append(img_path)

            # print(cur_all_imgs)
            # print(repr(prompt))

            action_ref = action_matching.action_2_format(step)

            response = generate_grounding(cur_all_imgs, prompt)
            # print(prompt)
            # print(cur_all_imgs)
            # print(response)
            response = process_string(response)
            
            num += 1
            
            try:
                action_pred = action_matching.pred_2_format(ast.literal_eval(response))
                annot_position = np.array(
                    [step["annot_position"][i:i + 4] for i in range(0, len(step["annot_position"]), 4)])
                check_match = action_matching.check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                  action_pred["action_type"], action_ref["touch_point"],
                                                                  action_ref["lift_point"], action_ref["action_type"],
                                                                  annot_position)
                # step accuracy
                if check_match == True:
                    corr_action += 1
                    match_label = 1
                    step_json['correct'] = 'yes'
                    # print("Step: " + str(j) + " right")
                else:
                    match_label = 0
                    # print("Step: " + str(j) + " wrong")

                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    corr_type += 1

                # text accuracy
                if action_ref["action_type"] == 3:
                    num_text += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        corr_text += 1

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if action_matching.is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        num_click += 1
                        if match_label:
                            corr_click += 1
                    # scroll accuracy
                    else:
                        num_scroll += 1
                        if match_label:
                            corr_scroll += 1
                    if (action_pred["action_type"] == 4) and action_matching.is_tap_action(action_ref["touch_point"],
                                                                                           action_ref[
                                                                                               "lift_point"]) and action_matching.is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        num_both_click += 1
                        if match_label:
                            corr_both_click += 1

            except:
                num_wrong_format += 1
                # print("Step: " + str(j) + " wrong format")
            all_save_results.append(step_json)
    score_average += corr_action / num

    print("Action Acc: " + str(corr_action / num))
    print("Type Acc: " + str(corr_type / num))
    print("Text Acc: " + str(corr_text / num_text))
    print("Click Acc: " + str(corr_click / num_click))
    print("Scroll Acc: " + str(corr_scroll / num_scroll))
    print("Both Click Acc: " + str(corr_both_click / num_both_click))
    print("Num Both Click: " + str(num_both_click))
    print("Num wrong format: " + str(num_wrong_format))
print("Average score: " + str(score_average / 5))
write_json(all_save_results, args.save_path)
