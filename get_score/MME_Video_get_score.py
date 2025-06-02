import json
import argparse
from collections import defaultdict

def calculate_accuracy(jsonl_path):
    # 初始化统计字典（支持嵌套统计）
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_entries = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                task_type = entry["task_type"]
                correct_answer = entry["answer"]
                predicted_answer = entry["filtered_resps"][0]  # 取第一个预测结果

                # 更新统计
                stats[task_type]["total"] += 1
                total_entries += 1
                if predicted_answer == correct_answer:
                    stats[task_type]["correct"] += 1
                    total_correct += 1
            except (KeyError, json.JSONDecodeError) as e:
                print(f"数据解析错误：{str(e)}")

    # 打印分类准确率
    print("分类准确率统计：")
    for task_type, data in stats.items():
        accuracy = data["correct"] / data["total"] if data["total"] else 0
        print(f"• {task_type.ljust(20)}: {accuracy:.2%} ({data['correct']}/{data['total']})")
    
    # 计算整体准确率
    overall_accuracy = total_correct / total_entries if total_entries else 0
    print(f"\n整体准确率：{overall_accuracy:.2%} ({total_correct}/{total_entries})")

if __name__ == "__main__":
    # 创建参数解析器[1,2](@ref)
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中各任务类型的准确率',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需的文件路径参数[3,5](@ref)
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='JSONL文件路径（包含评估数据）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用计算函数
    calculate_accuracy(args.jsonl_path)