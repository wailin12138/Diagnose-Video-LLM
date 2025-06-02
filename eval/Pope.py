import argparse
import json
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Qwen2.5-VL模型推理脚本 - POPE数据集处理',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入JSONL文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出JSONL文件路径')
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='图像基础目录路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--min_pixels', type=int, default=None,
                        help='最小像素值（默认：None）')
    parser.add_argument('--max_pixels', type=int, default=None,
                        help='最大像素值（默认：None）')
    parser.add_argument('--max_new_tokens', type=int, default=3,
                        help='生成的最大token数（默认：3）')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"加载模型: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # 配置处理器参数
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs['min_pixels'] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs['max_pixels'] = args.max_pixels
        
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    print("模型加载成功\n")

    with open(args.input_file, "r") as infile, open(args.output_file, "a") as outfile:
        for line in infile:
            obj = json.loads(line.strip())
            
            # 构建图像路径
            category = obj["doc"]["category"]
            img_path = os.path.join(
                args.image_base_dir, 
                category, 
                f"{obj['doc']['id']}.jpg"
            )
            
            # 构建问题文本
            question = f"{obj['doc']['question']} Please answer yes or no."
            
            print(f"处理图像: {img_path}")
            print(f"问题: {question}")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }]

            # 预处理输入
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # 生成答案
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            print(f"原始回答: {answer}")
            final_answer = "Yes" if "yes" in answer.lower() else "No"
            print(f"提取答案: {final_answer}")
            
            # 更新结果
            obj["resps"] = [[answer]]
            obj["filtered_resps"] = [final_answer]
            outfile.write(json.dumps(obj) + "\n")
            outfile.flush()

if __name__ == "__main__":
    main()