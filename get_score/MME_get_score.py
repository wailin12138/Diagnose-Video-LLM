import json
import argparse
from collections import defaultdict

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def calculate_scores(data):
    """计算每个类别的perception score和准确率"""
    categories = defaultdict(lambda: {
        'questions': [],
        'subtasks': defaultdict(list)
    })
    
    # 按类别和问题ID分组
    for entry in data:
        category = entry['doc']['category']
        question_id = entry['doc']['question_id']
        categories[category]['questions'].append(entry)
        categories[category]['subtasks'][question_id].append(entry)
    
    results = {}
    for category, info in categories.items():
        # 计算准确率（基于每个问题）
        correct_questions = 0
        total_questions = 0
        for question in info['questions']:
            total_questions += 1
            answer = question['filtered_resps'][0].strip().lower()
            target = question['target'].strip().lower()
            if answer == target:
                correct_questions += 1
        accuracy = correct_questions / total_questions if total_questions else 0
        
        # 计算accuracy_plus（基于每个完整子任务）
        correct_sub = 0
        total = 0
        for subtask_id, subtask_entries in info['subtasks'].items():
            if len(subtask_entries) != 2:
                continue  # 跳过不完整的子任务
            
            total += 1
            correct = 0
            # 计算子任务中的正确问题数
            for entry in subtask_entries:
                ans = entry['filtered_resps'][0].strip().lower()
                tgt = entry['target'].strip().lower()
                if ans == tgt:
                    correct += 1

            if correct == 2:
                correct_sub += 1

        accuracy_plus = correct_sub / total if total else 0
        
        results[category] = {
            'perception_score': accuracy * 100 + accuracy_plus * 100,
            'accuracy': accuracy
        }
    
    return results

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='计算MME感知和认知分数',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加输入文件参数
    parser.add_argument(
        'input_file',
        type=str,
        help='JSONL文件路径（包含评估数据）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 读取数据并计算分数
    data = read_jsonl(args.input_file)
    scores = calculate_scores(data)
    
    # 计算总分
    mme_perception_score = scores['existence']['perception_score'] + scores['color']['perception_score'] + \
                           scores['count']['perception_score'] + scores['position']['perception_score'] + \
                           scores['posters']['perception_score'] + scores['celebrity']['perception_score'] + \
                           scores['scene']['perception_score'] + scores['landmark']['perception_score'] + \
                           scores['artwork']['perception_score'] + scores['OCR']['perception_score']
    
    mme_cognition_score = scores['code_reasoning']['perception_score'] + scores['numerical_calculation']['perception_score'] + \
                          scores['text_translation']['perception_score'] + scores['commonsense_reasoning']['perception_score']
    
    # 根据输出格式打印结果
    if args.output == 'json':
        result_json = {
            "mme_perception_score": round(mme_perception_score, 4),
            "mme_cognition_score": round(mme_cognition_score, 4),
            "categories": {cat: {"perception_score": round(metrics['perception_score'], 4),
                                "accuracy": round(metrics['accuracy'], 4)}
                          for cat, metrics in scores.items()}
        }
        print(json.dumps(result_json, indent=2))
    else:
        print(f"mme_perception_score: {mme_perception_score:.4f}")
        print(f"mme_cognition_score: {mme_cognition_score:.4f}")
        for category, metrics in scores.items():
            print(f"\nCategory: {category}")
            print(f"  Perception Score: {metrics['perception_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.0%}")