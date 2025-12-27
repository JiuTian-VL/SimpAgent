import json
import os
import re

from PIL import Image, ImageDraw
from src.training.my_qwen_vl_utils_single import process_vision_info, process_vision_info_with_mask
from transformers import AutoProcessor
# from transformers import Qwen2VLForConditionalGeneration
from src.model_file.LLM_compression_v2_action.modeling_qwen2vl import Qwen2VLForConditionalGeneration
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

import pdb
import argparse
def write_json(data, file_path, task):
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, task + '.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/eval_results/",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data7/Users/zxr/hyh/SimpAgent-main/data/Qwen2-VL-finetune-code/Models/mindweb_bs16_lr5e-4_mask513_drop3_alpha1_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=2,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=3,
                    help='The path where the model is loaded from')
parser.add_argument('--task', type=str, default="domain",
                    help='The path where the model is loaded from')

# 解析参数
args = parser.parse_args()
args.save_path = os.path.join(args.save_path, args.model_path.split('/')[-1])


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
    
# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
    args.model_path, torch_dtype="auto", device_map="cuda"
    #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
# processor = AutoProcessor.from_pretrained("OS-Copilot/OS-Atlas-Base-7B")
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v2/checkpoint-1056/")
min_pixels = 200704
max_pixels = 1003520
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v5", min_pixels=min_pixels, max_pixels=max_pixels)
# processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/OS-Atlas-Base-7B/")
# processor = AutoProcessor.from_pretrained("/nas_sh/wentao/Qwen2-VL-7B-Instruct/")
processor = AutoProcessor.from_pretrained("/data1/Models/Qwen2-VL-2B-Instruct")

model.drop_k = args.drop_k
model.model.drop_k = args.drop_k

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



# evaluation on mind2web
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
from PIL import Image
import numpy as np

# logging.basicConfig(level=logging.INFO)


# convert action to prediction format (and return the groundtruth bbox)
def action2step(action, image_size, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{int(1000*item)}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step


# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

task_list = [args.task]
for args_task in task_list:

    mind2web_imgs_dir = "/data1/GUIData/seeclickdata/Mind2web/ming2web_images"
    mind2web_test = json.load(open('/data1/GUIData/seeclickdata/Mind2web/' + 'mind2web_data_test_' + args_task + '.json', 'r'))
    prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}\n"
    results = []
    for episode in tqdm(mind2web_test):
        goal = episode["confirmed_task"]
        annot_id = episode["annotation_id"]
        previous_actions = []
        results_actions = []
        previous_imgs = []

        for j, step in enumerate(episode["actions"]):
            if "bbox" not in step:
                print("action not found")
                continue

            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = Image.open(img_path)

            prompt = prompt_origin.format(goal)
            cur_all_imgs = []
            cur_step_idx = len(previous_imgs[-args.his_num:])

            previous_step = ""
            for i, action in enumerate(previous_actions[-args.his_num:]):
                    prompt += 'Image_' + str(i) + ":<image>\n"
                    prompt += 'Step_' + str(i) + ': ' + action + " .\n"
                    cur_all_imgs.append(previous_imgs[-args.his_num:][i])

            action_step = action2step(step, image.size)
            previous_actions.append(action_step)
            previous_imgs.append(img_path)
            cur_all_imgs.append(img_path)

            prompt += 'Image_' + str(cur_step_idx) + ":<image>\n"

            action_step_ref, bbox_ref = action2step(step, image.size, return_bbox=True)
            try:
                action_step_ref = ast.literal_eval(action_step_ref)
            except:
                continue

            response = generate_grounding(cur_all_imgs, prompt)
            response = process_string(response)
            # print(response)


            step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                        "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}
            try:
                action_pred = ast.literal_eval(response)

                if action_pred["action_type"] == action_step_ref["action_type"]:
                    step_result["Op_match"] = True

                click_point = action_pred["click_point"]

                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True

                # 按照mind2web的方式，把action转换成一个字符串，即如果是TYPE需要考虑字符间的F1
                pred_str = str(action_pred["action_type"])
                if action_pred["action_type"] == 3 or action_pred["action_type"] == 2:
                    pred_str += ' '
                    pred_str += action_pred["value"].lower()
                ref_str = str(action_step_ref["action_type"])
                if action_step_ref["action_type"] == 3 or action_step_ref["action_type"] == 2:
                    ref_str += ' '
                    ref_str += action_step_ref["value"].lower()

                op_f1 = calculate_f1(pred_str, ref_str)
                step_result["Op_F1"][0] = op_f1

            except:
                logging.info("format wrong")

            # logging.info(step_result)

            results_actions.append(step_result)    

        results.append(results_actions)


    # calculate metrics
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {4: [], 2: [], 3: []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0
    for i, item in enumerate(results):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True
        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])

    logging.info("Operation F1: " + str(marco_op_f1))
    logging.info("Element Acc: " + str(num_ele / num_step))
    logging.info("Step Success: " + str(num_step_success / num_step))
    logging.info("Episode Success: " + str(num_episode_success / num_episode))
    logging.info("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

    macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])
    logging.info("Macro Ele Acc: " + str(macro_ele_acc))
    logging.info("Macro Op F1: " + str(macro_action_f1))
    logging.info("Macro Step SR: " + str(macro_step_acc))

    results = {
        "Operation_F1": float(marco_op_f1),
        "Element_Accuracy": float(num_ele / num_step) if num_step != 0 else 0.0,
        "Step_Success_Rate": float(num_step_success / num_step) if num_step != 0 else 0.0,
        "Episode_Success_Rate": float(num_episode_success / num_episode) if num_episode != 0 else 0.0,
        "Operation_F1_Categories": [float(np.mean(x)) for x in op_f1.values()],
        "Macro_Element_Accuracy": float(macro_ele_acc),
        "Macro_Operation_F1": float(macro_action_f1),
        "Macro_Step_Success_Rate": float(macro_step_acc)
    }
    print(args_task)
    print(results)


    write_json(results, args.save_path, args_task)