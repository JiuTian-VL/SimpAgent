import json
import re
# visualize&process aitw data
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
import random
import os
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', default="/data7/Users/zxr/zhouxurui/GUI-dataset/aitw_seeclick/aitw_images", type=str)
args = parser.parse_args()


# show image with point
def show_image_with_point(image, point=None):

    img_width, img_height = image.size
    dpi = 40
    figsize = img_width / float(dpi), img_height / float(dpi)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if point:
        x = int(point[0]*img_width)
        y = int(point[1]*img_height)
        ax.plot(x, y, marker='o', markersize=10, color='r')  # 绘制红色圆点
    plt.axis('off')
    plt.show()


# convert action to prediction format
def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [int(1000 * (touch_point[0] + lift_point[0]) / 2), int(1000 * (touch_point[1] + lift_point[1]) / 2)]
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


aitw_imgs_dir = args.imgs_dir
aitw_train = json.load(open('aitw_data_train.json', 'r'))
aitw_train = aitw_train["general"] + aitw_train["single"] + aitw_train["webshopping"] + \
                  aitw_train["install"] + aitw_train["googleapps"]

# aitw_train = aitw_train["googleapps"]


train_step = []
prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
step_i = 0
for episode in tqdm(aitw_train):

    previous_actions = []
    previous_imgs = []

    for step in episode:
        img_filename = step["img_filename"] + '.png'
        img_path = os.path.join(aitw_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print('image not found')
            continue
        # if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
        #     continue
        image = Image.open(img_path)

        goal = step["goal"]

        prompt = prompt_origin.format(goal)

        cur_step_preimg = previous_imgs[-4:]
        cur_step_idx = len(previous_imgs[-4:])
        cur_all_imgs = []
        for i, action in enumerate(previous_actions[-4:]):
            prompt += 'Image_' + str(i) + ":<image>\n\n"
            prompt += 'Step_' + str(i) + ':' + action + " .\n"
            cur_all_imgs.append(previous_imgs[-4:][i])

        action_step = action2step(step)
        previous_actions.append(action_step)
        previous_imgs.append(img_path)

        # visualize step data
        # action_step_vis = ast.literal_eval(action_step)
        # point = action_step_vis['click_point'] if 'click_point' in action_step_vis else None
        # show_image_with_point(image, point)
        # print(step)
        # input()
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
json.dump(train_step, open("aitw_train_sft_all_999_sequence_llavaformat.json", "w"))