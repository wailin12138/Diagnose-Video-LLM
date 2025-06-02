import json
import argparse

def calculate_accuracy(file_path):
    """计算JSONL文件中预测答案的准确率"""
    correct = 0
    total = 0

    with open(file_path, 'r', encoding='utf-8') as file:  # 添加编码处理[1](@ref)
        for line in file:
            data = json.loads(line)
            correct_answer = data['correct_answer']
            predicted_answer = data['filtered_resps'][0]  # 提取首层列表的第一个元素

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, total, correct

if __name__ == "__main__":
    # 创建参数解析器[2,3](@ref)
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中预测答案的准确率',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需的文件路径参数[5,7](@ref)
    parser.add_argument(
        'file_path',
        type=str,
        help='JSONL文件路径（包含正确答案和预测结果）'
    )
    
    # 解析命令行参数[3,6](@ref)
    args = parser.parse_args()
    
    # 计算准确率
    accuracy, total, correct = calculate_accuracy(args.file_path)
    
    # 输出结果
    if args.verbose:
        print(f"详细统计:")
        print(f"• 正确预测数: {correct}")
        print(f"• 总样本数: {total}")
        print(f"• 准确率: {accuracy:.2f}%")
    else:
        print(f"Accuracy: {accuracy:.2f}%")