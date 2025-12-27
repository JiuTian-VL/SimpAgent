import json
import os
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', default="/data1/GUIData/AndroidControl/androidcontrol_images/", type=str)
args = parser.parse_args()

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

def check_image_exists(img_path):
    """检查图片是否存在且有效"""
    if not os.path.exists(img_path):
        return False, "File not exists"
    
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {e}"

def process_androidcontrol_data():
    """处理AndroidControl训练数据，生成两种格式的JSON文件"""
    
    # 文件路径
    androidcontrol_train_path = "/data7/Users/zxr/hyh/SimpAgent-main/data/AndroidControl/androidcontrol_data_train.json"
    androidcontrol_imgs_dir = args.imgs_dir
    
    # 输出文件路径
    output_path_compact = "androidcontrol_train_llavaformat_no_finish_add_longpress.json"  # 压缩格式
    output_path_pretty = "androidcontrol_train_llavaformat_no_finish_add_longpress_pretty.json"  # 格式化版本
    
    try:
        # 1. 读取AndroidControl训练数据
        with open(androidcontrol_train_path, 'r', encoding='utf-8') as f:
            androidcontrol_data = json.load(f)
        
        print(f"成功加载AndroidControl训练数据，共{len(androidcontrol_data)}个episode")
        
        # 2. 处理每个episode
        train_step = []
        prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
        step_i = 0
        episode_count = 0
        valid_steps = 0
        invalid_steps = 0
        action_type_stats = {}  # 统计动作类型
        
        for episode in tqdm(androidcontrol_data, desc="Processing episodes"):
            episode_count += 1
            
            # 提取episode信息
            episode_id = episode.get("episode_id", episode_count)
            actions = episode.get("actions", [])
            step_instructions = episode.get("step_instructions", [])
            goal = episode.get("goal", "")
            images = episode.get("images", [])
            screenshot_widths = episode.get("screenshot_widths", [])
            screenshot_heights = episode.get("screenshot_heights", [])
            
            # 确保每个步骤都有对应的信息
            num_steps = min(len(actions), len(step_instructions), len(images), len(screenshot_widths), len(screenshot_heights))
            if num_steps == 0:
                continue
            
            previous_actions = []
            previous_imgs = []
            
            # 处理每个步骤
            for step_idx in range(num_steps):
                # 获取当前步骤的图片路径
                img_filename = images[step_idx]
                img_path = os.path.join(androidcontrol_imgs_dir, img_filename)
                
                # 检查图片文件是否存在且有效
                img_exists, img_status = check_image_exists(img_path)
                if not img_exists:
                    invalid_steps += 1
                    if invalid_steps <= 10:  # 只显示前10个无效图片的警告
                        print(f"跳过无效图片: {img_path} - {img_status}")
                    continue
                
                # 获取当前步骤的图片尺寸
                img_width = screenshot_widths[step_idx] if step_idx < len(screenshot_widths) else 0
                img_height = screenshot_heights[step_idx] if step_idx < len(screenshot_heights) else 0
                
                # 如果尺寸无效，尝试从图片文件获取实际尺寸
                if img_width <= 0 or img_height <= 0:
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                    except:
                        img_width, img_height = 1080, 2400  # 使用默认尺寸
                
                # 打开图片
                image = Image.open(img_path)
                
                # 构建提示词
                prompt = prompt_origin.format(goal)
                
                # 添加历史信息（最多2步）
                cur_step_preimg = previous_imgs[-2:]
                cur_step_idx = len(cur_step_preimg)
                cur_all_imgs = []
                
                for i, action in enumerate(previous_actions[-2:]):
                    prompt += 'Image_' + str(i) + ":<image>\n\n"
                    prompt += 'Step_' + str(i) + ':' + action + " .\n"
                    if i < len(cur_step_preimg):
                        cur_all_imgs.append(cur_step_preimg[i])
                
                # 转换当前动作，传入图片尺寸用于归一化
                action_str = actions[step_idx]
                action_step = android_action2step(action_str, img_width, img_height)
                
                # 统计动作类型
                try:
                    action_data = json.loads(action_str)
                    action_type = action_data.get("action_type", "unknown")
                    if action_type not in action_type_stats:
                        action_type_stats[action_type] = 0
                    action_type_stats[action_type] += 1
                except:
                    pass
                
                # 更新历史记录
                previous_actions.append(action_step)
                previous_imgs.append(img_path)
                
                # 构建对话格式
                conversations = []
                conv_user = {"value": "", "from": "human"}
                conv_user["value"] += prompt
                conv_user["value"] += 'Image_' + str(cur_step_idx) + ":<image>\n\n"
                conv_ai = {"value": str(action_step), "from": "assistant"}
                conversations.append(conv_user)
                conversations.append(conv_ai)
                cur_all_imgs.append(img_path)
                
                # 创建样本数据并添加到列表
                train_step.append({
                    "conversations": conversations, 
                    "image": cur_all_imgs
                })
                
                step_i += 1
                valid_steps += 1
        
        # 3. 保存压缩版本（单行，无换行）
        with open(output_path_compact, 'w', encoding='utf-8') as out_file:
            json.dump(train_step, out_file, ensure_ascii=False, separators=(',', ':'))
        
        # 4. 保存格式化版本（带缩进和换行，便于观察）
        with open(output_path_pretty, 'w', encoding='utf-8') as out_file:
            json.dump(train_step, out_file, ensure_ascii=False, indent=2)
        
        # 5. 输出统计信息
        print(f"\n处理完成:")
        print(f"总episode数: {episode_count}")
        print(f"总步骤数: {step_i}")
        print(f"有效步骤数: {valid_steps}")
        print(f"无效步骤数（图片问题）: {invalid_steps}")
        
        print(f"\n动作类型统计:")
        for action_type, count in sorted(action_type_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / valid_steps) * 100
            print(f"  {action_type}: {count}次 ({percentage:.2f}%)")
        
        print(f"\n文件保存:")
        print(f"压缩版本: {output_path_compact}")
        print(f"格式化版本: {output_path_pretty}")
        
        # 6. 验证输出文件
        print(f"\n验证输出文件:")
        
        # 验证压缩版本
        with open(output_path_compact, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"压缩版本文件大小: {len(content)} 字符")
            print(f"压缩版本是否只有一行: {content.count(chr(10)) == 0}")
            
            try:
                loaded_data = json.loads(content)
                print(f"压缩版本成功加载JSON，包含 {len(loaded_data)} 个样本")
            except Exception as e:
                print(f"压缩版本JSON验证失败: {e}")
        
        # 验证格式化版本
        with open(output_path_pretty, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"格式化版本文件大小: {len(content)} 字符")
            print(f"格式化版本行数: {content.count(chr(10)) + 1}")
            
            try:
                loaded_data = json.loads(content)
                print(f"格式化版本成功加载JSON，包含 {len(loaded_data)} 个样本")
                
                # 显示第一个样本的详细信息
                if len(loaded_data) > 0:
                    print(f"\n第一个样本的详细信息:")
                    print(f"键: {list(loaded_data[0].keys())}")
                    
                    if "conversations" in loaded_data[0] and len(loaded_data[0]["conversations"]) > 1:
                        user_prompt = loaded_data[0]["conversations"][0]["value"]
                        assistant_response = loaded_data[0]["conversations"][1]["value"]
                        
                        print(f"用户提示长度: {len(user_prompt)} 字符")
                        print(f"用户提示前200字符: {user_prompt[:200]}...")
                        print(f"助手响应: {assistant_response}")
                    
                    if "img_width" in loaded_data[0] and "img_height" in loaded_data[0]:
                        print(f"图片尺寸: {loaded_data[0]['img_width']}x{loaded_data[0]['img_height']}")
                    
                    if "image" in loaded_data[0]:
                        print(f"包含图片数量: {len(loaded_data[0]['image'])}")
                        for i, img_path in enumerate(loaded_data[0]['image']):
                            print(f"  图片{i}: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"格式化版本JSON验证失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return False

if __name__ == "__main__":
    # 只处理训练集
    process_androidcontrol_data()