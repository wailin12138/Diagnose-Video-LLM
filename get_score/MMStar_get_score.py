import json
import argparse
from collections import defaultdict

def calculate_accuracy(jsonl_path):
    # 初始化统计字典
    category_stats = {}
    l2_category_stats = {}
    overall_correct = 0
    overall_total = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            # 提取正确答案和预测答案
            correct_answer = entry['answer'].strip().upper()
            try:
                predicted_answer = entry['filtered_resps'][0].strip().upper()
            except (AttributeError, IndexError) as e:
                print(f"Error in doc_id={entry['index']}: {e}")
                predicted_answer = ""  # 设为默认值
            is_correct = (predicted_answer == correct_answer)

            # 提取分类信息
            category = entry['category']
            l2_category = entry['l2_category']

            # 更新category统计
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}
            category_stats[category]['total'] += 1
            if is_correct:
                category_stats[category]['correct'] += 1

            # 更新l2_category统计
            if l2_category not in l2_category_stats:
                l2_category_stats[l2_category] = {'correct': 0, 'total': 0}
            l2_category_stats[l2_category]['total'] += 1
            if is_correct:
                l2_category_stats[l2_category]['correct'] += 1

            # 更新整体统计
            overall_total += 1
            if is_correct:
                overall_correct += 1

    # 打印结果
    print("分类准确率（按大类 category）:")
    for cat, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] else 0
        print(f"{cat}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")

    print("\n分类准确率（按子类 l2_category）:")
    for l2_cat, stats in l2_category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] else 0
        print(f"{l2_cat}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\n整体准确率: {overall_correct / overall_total:.4f} ({overall_correct}/{overall_total})")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中各分类层级的准确率统计',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需的文件路径参数
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='JSONL文件路径（包含评估数据）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用计算函数
    calculate_accuracy(args.jsonl_path)