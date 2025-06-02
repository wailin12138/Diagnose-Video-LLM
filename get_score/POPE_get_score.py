import json
import argparse

def calculate_metrics(file_path):
    """计算分类指标（准确率、精确率、召回率、F1分数）"""
    y_true = []
    y_pred = []
    
    with open(file_path, 'r', encoding='utf-8') as f:  # 添加编码处理
        for line in f:
            data = json.loads(line)
            
            # 获取真实标签和预测结果
            target = data["target"].strip().lower()
            pred = data["filtered_resps"][0].strip().lower()
            
            # 统一处理标点符号
            if pred.endswith(('.', '?', '!')):
                pred = pred[:-1]
            
            # 转换为二元标签（yes=1, no=0）
            y_true.append(1 if target == "yes" else 0)
            y_pred.append(1 if pred == "yes" else 0)
    
    # 计算分类指标
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "support": len(y_true),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn
        }
    }

if __name__ == "__main__":
    # 创建参数解析器 [1,3,7](@ref)
    parser = argparse.ArgumentParser(
        description='计算JSONL文件中的分类指标（准确率、精确率、召回率、F1分数）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加必需的文件路径参数 [4,8](@ref)
    parser.add_argument(
        'file_path',
        type=str,
        help='JSONL文件路径（包含评估数据）'
    )
    
    # 解析命令行参数 [1,7](@ref)
    args = parser.parse_args()
    
    # 计算指标
    metrics = calculate_metrics(args.file_path)
    
    # 根据输出格式打印结果
    if args.output_format == 'json':
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Total Samples: {metrics['support']}")
        
        if args.verbose:
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"True Positives (TP): {cm['true_positive']}")
            print(f"False Positives (FP): {cm['false_positive']}")
            print(f"True Negatives (TN): {cm['true_negative']}")
            print(f"False Negatives (FN): {cm['false_negative']}")