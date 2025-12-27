import json
import re
from src.utils import text_matching
import math

data = json.load(open('/data7/Users/zxr/hyh/SimpAgent-main/odssey_eval/guiodssey_bs64_lr3e-5_mask535_drop6_alpha1_eval/odssey_results.json', 'r'))

gt_test = []
for item in data:
    gt_test.append({
        "pred": item["pred"],
        "gt": item["gt"]
    })

# 初始化统计变量
total_steps = len(gt_test)
format_wrong = 0
type_correct = 0
step_correct = 0
grounding_correct = 0
num_grounding = 0
type_text_acc = 0
type_text_cnt = 0
type_text_match_cnt = 0

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# 新增：每种 action_type 的统计
type_stats = {}  # 格式: {action_type: {"total": 0, "correct": 0, "accuracy": 0.0}}

def preprocess_json_string(s):
    """
    预处理 JSON 字符串，处理非标准格式
    """
    # 处理 click_point: (x,y) 格式
    # 匹配 "click_point": (数字,数字)
    s = re.sub(r'\"click_point\":\s*\((\d+),\s*(\d+)\)', r'"click_point": [\1, \2]', s)
    
    # 处理可能存在的其他非标准格式
    # 如果已经是字符串格式的 "(x,y)"，去掉引号
    s = re.sub(r'\"click_point\":\s*\"\((\d+),\s*(\d+)\)\"', r'"click_point": [\1, \2]', s)
    
    return s

for record in gt_test:
    try:
        pred_dict = json.loads(preprocess_json_string(record["pred"]))
        gt_dict = json.loads(preprocess_json_string(record["gt"]))
    except json.JSONDecodeError:
        format_wrong += 1
        continue
    
    # 获取 gt 的 action_type
    gt_action_type = gt_dict.get("action_type")
    
    # 初始化该 action_type 的统计（如果不存在）
    if gt_action_type not in type_stats:
        type_stats[gt_action_type] = {"total": 0, "correct": 0, "accuracy": 0.0}
    
    # 增加该类型的总数
    type_stats[gt_action_type]["total"] += 1
    
    # 检查 action_type 是否匹配
    type_match = pred_dict.get("action_type") == gt_action_type
    if type_match:
        type_correct += 1
        # 增加该类型的正确数
        type_stats[gt_action_type]["correct"] += 1
    else:
        # print(pred_dict.get("action_type"))
        # print(gt_dict.get("action_type"))
        continue
    
    # 检查 step 中其他键的匹配
    step_match = True
    grounding_match = None  # 仅对 click_point 进行统计

    for key in pred_dict:
        if key == "action_type":
            continue
        
        if key not in gt_dict:
            step_match = False
            break
        
        pred_value = pred_dict[key]
        gt_value = gt_dict[key]
        
        if key == "click_point":
            # 解析点击点为数值
            # 注意：这里假设 pred_value 和 gt_value 已经是列表格式 [x, y]
            # 如果不是列表格式，需要先转换
            if isinstance(pred_value, list) and isinstance(gt_value, list):
                # 计算欧几里得距离
                distance = math.sqrt((pred_value[0] - gt_value[0])**2 + (pred_value[1] - gt_value[1])**2)
                if distance <= 140:
                    matching = True
                    grounding_correct += 1  # 统计 grounding_acc
                else:
                    matching = False
                num_grounding += 1  # 总计数
                
                grounding_match = matching
            else:
                # 如果不是列表格式，记录错误
                print(f"警告: click_point 不是列表格式，pred={pred_value}, gt={gt_value}")
                step_match = False
                break
        elif key == "typed_text":
            if calculate_f1_score(pred_value, gt_value)>0.5:
                matching = True
                type_text_match_cnt += 1
            else:
                matching = False
            type_text_cnt += 1
        else:
            # 其他关键字必须完全匹配
            matching = pred_value == gt_value
        
        if not matching:
            step_match = False
            break
    
    # 统计 step_acc
    if step_match:
        step_correct += 1

# 计算每种 action_type 的准确率
for action_type in type_stats:
    total = type_stats[action_type]["total"]
    correct = type_stats[action_type]["correct"]
    if total > 0:
        type_stats[action_type]["accuracy"] = correct / total

# 计算总体准确率
type_acc = type_correct / (total_steps) if (total_steps) != 0 else 0
type_text_acc = type_text_match_cnt / type_text_cnt if type_text_cnt > 0 else 0
step_acc = step_correct / (total_steps) if (total_steps) != 0 else 0
grounding_acc = grounding_correct / num_grounding if num_grounding != 0 else 0

# 输出结果
print("=" * 50)
print(f"total steps: {total_steps}")
print(f"format_wrong: {format_wrong}")
print(f"type_acc: {type_acc:.3f}")
print(f"type_text_acc: {type_text_acc:.3f}")
print(f"step_acc: {step_acc:.3f}")
print(f"grounding_acc: {grounding_acc:.3f}")
print("=" * 50)

# 输出每种 action_type 的统计
print("\n每种 action_type 的统计:")
print("-" * 50)
print(f"{'Action Type':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
print("-" * 50)

# 按 action_type 排序输出
for action_type in sorted(type_stats.keys()):
    stats = type_stats[action_type]
    print(f"{action_type:<10} {stats['total']:<8} {stats['correct']:<8} {stats['accuracy']:<10.3f}")

print("-" * 50)

# 如果需要，也可以输出预测 action_type 的统计
print("\n预测 action_type 的分布统计:")
print("-" * 50)

pred_type_counts = {}
for record in gt_test:
    try:
        pred_dict = json.loads(preprocess_json_string(record["pred"]))
        pred_action_type = pred_dict.get("action_type")
        if pred_action_type is not None:
            pred_type_counts[pred_action_type] = pred_type_counts.get(pred_action_type, 0) + 1
    except:
        continue

print(f"{'Predicted Type':<15} {'Count':<8} {'Percentage':<10}")
print("-" * 50)
for action_type in sorted(pred_type_counts.keys()):
    count = pred_type_counts[action_type]
    percentage = count / total_steps if total_steps > 0 else 0
    print(f"{action_type:<15} {count:<8} {percentage:<10.3f}")
print("-" * 50)