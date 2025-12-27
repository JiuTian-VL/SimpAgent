# visualize&process mind2web data
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
import os
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', default="/data1/GUIData/seeclickdata/Mind2web/ming2web_images", type=str)
args = parser.parse_args()


# show image with bbox
def show_image_with_bbox(image, bbox=None):

    img_width, img_height = image.size
    dpi = 40
    figsize = img_width / float(dpi), img_height / float(dpi)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bbox:
        x = int(bbox['x'])
        y = int(bbox['y'])
        width = int(bbox['width'])
        height = int(bbox['height'])
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


# convert action to prediction format
def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']  # five types of data

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    
    # 修改：归一化后乘以1000取整（与aitw保持一致）
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [int(1000 * item) for item in click_point]  
    click_point = "({},{})".format(click_point[0], click_point[1])

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point, select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point, typed_text)
    return action_step


mind2web_imgs_dir = args.imgs_dir
mind2web_train = json.load(open('/data1/GUIData/seeclickdata/Mind2web/mind2web_data_train.json', 'r'))
train_step = []
prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
step_i = 0

for episode in tqdm(mind2web_train):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []
    previous_imgs = []

    for step in episode["actions"]:
        # Few actions can not find its corresponding bbox, jump these actions
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

        cur_step_preimg = previous_imgs[-2:]
        cur_step_idx = len(previous_imgs[-2:])
        cur_all_imgs = []
        
        for i, action in enumerate(previous_actions[-2:]):
            prompt += 'Image_' + str(i) + ":<image>\n\n"
            prompt += 'Step_' + str(i) + ':' + action + " .\n"
            cur_all_imgs.append(previous_imgs[-2:][i])
        action_step = action2step(step, image.size)

        previous_actions.append(action_step)
        previous_imgs.append(img_path)

        conversations = []
        conv_user = {"value": "", "from": "human"}
        conv_user["value"] += prompt
        conv_user["value"] += 'Image_' + str(cur_step_idx) + ":<image>\n\n"
        conv_ai = {"value": str(action_step), "from": "assistant"}
        conversations.append(conv_user)
        conversations.append(conv_ai)

        cur_all_imgs.append(img_path)

        train_step.append({"conversations": conversations, "image": cur_all_imgs})
        step_i += 1

# random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open("mind2web_train_sft_all_sequence_llavaformat.json", "w"))