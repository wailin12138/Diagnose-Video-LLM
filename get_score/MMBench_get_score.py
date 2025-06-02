import json
import argparse

def calculate_accuracy(jsonl_path):
    # 初始化统计字典
    category_stats = {}
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
                print(f"Error in index={entry['index']}: {e}")
                predicted_answer = ""  # 设为默认值
            is_correct = (predicted_answer == correct_answer)

            # 提取分类信息
            category = entry['task_type']

            # 更新category统计
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}
            category_stats[category]['total'] += 1
            if is_correct:
                category_stats[category]['correct'] += 1

            # 更新整体统计
            overall_total += 1
            if is_correct:
                overall_correct += 1

    # 打印结果
    print("分类准确率:")
    for cat, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] else 0
        print(f"{cat}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\n整体准确率: {overall_correct / overall_total:.4f} ({overall_correct}/{overall_total})")

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中答案的准确率统计',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需的位置参数
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='JSONL文件路径（包含答案和预测结果）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用计算函数
    calculate_accuracy(args.jsonl_path)