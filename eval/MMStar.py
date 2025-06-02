import argparse
import json
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_answer(text):
    """从文本中精确提取大写字母A-D"""
    clean_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    patterns = [
        re.compile(r'(?:答案|正确选项|正确答案|选项|정답|선택지|옵션)[:：]?\s*([A-D])', re.I),
        re.compile(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])')
    ]
    for pattern in patterns:
        match = pattern.search(clean_text)
        if match and len(match.groups()) >= 1:
            candidate = match.group(1).upper()
            if candidate in {'A', 'B', 'C', 'D'}:
                return candidate
    final_match = re.search(r'\b([A-D])\b', clean_text, re.I)
    return final_match.group(1).upper() if final_match else None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视觉语言模型推理脚本')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='输入JSONL文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出JSONL文件路径')
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='图像基础目录路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--max_new_tokens', type=int, default=1,
                        help='生成的最大token数（默认：1）')
    parser.add_argument('--min_pixels', type=int, default=None,
                        help='最小像素值（默认：None）')
    parser.add_argument('--max_pixels', type=int, default=None,
                        help='最大像素值（默认：None）')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"加载模型: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # 根据参数配置处理器
    processor_kwargs = {}
    if args.min_pixels is not None and args.max_pixels is not None:
        processor_kwargs['min_pixels'] = args.min_pixels
        processor_kwargs['max_pixels'] = args.max_pixels
    
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    print("模型加载成功\n")

    with open(args.input_file, "r") as infile, open(args.output_file, "a") as outfile:
        for line in infile:
            obj = json.loads(line.strip())
            meta_info = obj['meta_info']
            img_path = os.path.join(args.image_base_dir, meta_info['image_path'])
            question = obj["question"] + "\n" + "Please answer the letter of the option directly in the given option."

            print(f"处理图像: {img_path}")
            print(f"问题: {question}")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }]

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
            final_answer = extract_answer(answer)
            print(f"提取答案: {final_answer}")
            
            obj["resps"] = [[answer]]
            obj["filtered_resps"] = [final_answer]
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            outfile.flush()

if __name__ == "__main__":
    main()