import os, json
import argparse
import numpy as np
import random
from tqdm import tqdm


DATA_DIR = "/data1/GUIData/GUI-Odyssey/data"
pic_base = os.path.join(DATA_DIR, 'screenshots')
anno_base = os.path.join(DATA_DIR, 'annotations')
output_dir = os.path.join(DATA_DIR, '/data7/Users/zxr/hyh/SimpAgent-main/data/guiodssey/')

# Use the same prompt format as AITW
PROMPT_ORIGIN = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"

def decode_action(action, info):
    """Convert action to AITW format JSON string"""
    if action == 'CLICK' or action == "LONG_PRESS":
        if info == 'KEY_HOME':
            action_type = 6  # PRESS_HOME
            action_json = '{"action_type": 6}'
        elif info == 'KEY_BACK':
            action_type = 5  
            action_json = '{"action_type": 5}'
        elif info == 'KEY_APPSELECT':
            action_type = 2  
            action_json = '{"action_type": 2}'
        elif type(info) == list:
            click_point = [int(info[0][0]), int(info[0][1])]
            click_point_str = "({},{})".format(click_point[0], click_point[1])
            if action == 'CLICK':
                action_type = 4  # DUAL_POINT
            elif action == "LONG_PRESS":
                action_type = 12
            action_json = '{{"action_type": {}, "click_point": {}}}'.format(action_type, click_point_str)
        else:
            raise ValueError(f'Unknown click action {info}')

    elif action == 'SCROLL':
        start = np.array(info[0])
        end = np.array(info[1])
        delta = end - start
        delta_abs = np.abs(delta)
        
        if delta_abs[0] > delta_abs[1]:
            action_type = 9 if delta[0] > 0 else 8  # RIGHT/LEFT
        else:
            action_type = 1 if delta[1] < 0 else 0  # UP/DOWN
            
        action_json = '{{"action_type": {}}}'.format(action_type)
    elif action == 'TEXT': 
        action_type = 3  # TYPE
        action_json = '{{"action_type": {}, "typed_text": "{}"}}'.format(action_type, info)
    elif action == 'COMPLETE':
        action_type = 10  # STATUS_TASK_COMPLETE
        action_json = '{{"action_type": {}}}'.format(action_type)
    elif action == 'INCOMPLETE':
        action_type = 11  # STATUS_TASK_IMPOSSIBLE
        action_json = '{{"action_type": {}}}'.format(action_type)
    else:
        raise ValueError(f'Unknown action {action}')
    return action_json


def get_all_annotation_files():
    """Get all JSON files in the annotations directory"""
    if not os.path.exists(anno_base):
        raise FileNotFoundError(f"Annotations directory does not exist: {anno_base}")
    
    annotation_files = []
    for file in os.listdir(anno_base):
        if file.endswith('.json'):
            annotation_files.append(file)
    
    print(f"Found {len(annotation_files)} annotation files")
    return annotation_files

def load_split_file(split_file_path):
    """Load train/test split from split file"""
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file does not exist: {split_file_path}")
    
    with open(split_file_path, 'r') as f:
        split_data = json.load(f)
    
    # 确保split文件包含train和test键
    if 'train' not in split_data or 'test' not in split_data:
        raise ValueError("Split file must contain 'train' and 'test' keys")
    
    train_files = set(split_data['train'])
    test_files = set(split_data['test'])
    
    print(f"Loaded split file: {len(train_files)} train files, {len(test_files)} test files")
    
    return train_files, test_files

def display_samples(data_list, title, num_samples=5):
    """Display sample conversations"""
    print("\n" + "="*80)
    print(f"{title} (showing first {min(num_samples, len(data_list))} samples):")
    print("="*80)
    
    for i, sample in enumerate(data_list[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        print("-" * 40)



def build_standard_dataset(his_len=4, instr_level='high'):
    """Build standard training dataset, split into train and test sets"""
    os.makedirs(output_dir, exist_ok=True)
    annotation_files = get_all_annotation_files()
    
    # 加载划分文件
    split_file_path = os.path.join(DATA_DIR, 'splits', 'random_split.json')
    train_files_set, test_files_set = load_split_file(split_file_path)
    
    train_data = []
    test_data = []
    
    for file in tqdm(annotation_files, desc="Processing annotation files"):
        file_path = os.path.join(anno_base, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            high_level_instruction = data['task_info']['instruction']
            steps = data['steps']
            
            previous_actions = []
            previous_imgs = []

            for step in steps:
                image = step['screenshot']
                action = step['action']
                info = step['info']
                
                if instr_level == 'high':
                    instruction = high_level_instruction
                    
                img_abs_path = os.path.join(pic_base, image)
                
                # Build the same prompt format as AITW
                prompt = PROMPT_ORIGIN.format(instruction)
                
                # Add historical actions and images (up to his_len)
                cur_step_preimg = previous_imgs[-his_len:]
                cur_step_idx = len(previous_imgs[-his_len:])
                cur_all_imgs = []
                
                for i, action_str in enumerate(previous_actions[-his_len:]):
                    prompt += 'Image_' + str(i) + ":<image>\n\n"
                    prompt += 'Step_' + str(i) + ':' + action_str + " .\n"
                    cur_all_imgs.append(previous_imgs[-his_len:][i])
                
                # Convert action to AITW format
                action_step = decode_action(action, info)
                previous_actions.append(action_step)
                previous_imgs.append(img_abs_path)
                
                # Build conversation format (exactly the same as AITW)
                conversations = []
                conv_user = {"value": "", "from": "human"}
                conv_user["value"] += prompt
                conv_user["value"] += 'Image_' + str(cur_step_idx) + ":<image>\n\n"
                conv_ai = {"value": str(action_step), "from": "assistant"}
                conversations.append(conv_user)
                conversations.append(conv_ai)
                cur_all_imgs.append(img_abs_path)
                
                # 根据划分文件将数据分配到训练集或测试集
                sample = {
                    "conversations": conversations, 
                    "image": cur_all_imgs
                }
                
                if file in train_files_set:
                    train_data.append(sample)
                elif file in test_files_set:
                    test_data.append(sample)
                else:
                    # 有些文件可能不在划分文件中，我们可以忽略或分配到默认集合
                    print(f"Warning: File {file} not found in split file, adding to train set")
                    train_data.append(sample)
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    # 显示示例
    display_samples(train_data, "Training Set Samples")
    display_samples(test_data, "Test Set Samples")
    
    # Shuffle data randomly
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Save train and test datasets separately
    train_filename = f"guiodyssey_standard_train_{instr_level}_llavaformat.json"
    test_filename = f"guiodyssey_standard_test_{instr_level}_llavaformat.json"
    
    train_output_path = os.path.join(output_dir, train_filename)
    test_output_path = os.path.join(output_dir, test_filename)
    
    # Save train set
    with open(train_output_path, 'w') as f:
        json.dump(train_data, f, indent=None, separators=(',', ':'), ensure_ascii=False)
    
    # Save test set
    with open(test_output_path, 'w') as f:
        json.dump(test_data, f, indent=None, separators=(',', ':'), ensure_ascii=False)
    
    print(f"\nTrain dataset saved: {train_output_path}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test dataset saved: {test_output_path}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, test_data

def main(args):
    """Main function"""
    print(f"Starting data processing, history length: {args.his_len}, instruction level: {args.level}")
    
    # Verify if data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")
    
    if args.type == 'standard':
        train_data, test_data = build_standard_dataset(args.his_len, args.level)
    
    print("Data processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GUI Odyssey dataset")
    parser.add_argument('--his_len', type=int, default=4, help='History length')
    parser.add_argument('--level', type=str, choices=['high', 'low'], default='high', help='Instruction level')
    parser.add_argument('--type', type=str, choices=['semantic', 'standard'], default='standard', 
                       help='Data type: standard-standard format, semantic-semantic enhanced format')
    args = parser.parse_args()
    
    main(args)