import json
import os
import sys
import boto3
from api_request_schema import api_request_list, get_model_ids
import pytesseract
from PIL import Image
from langdetect import detect

# 基础配置
model_id = os.getenv('MODEL_ID', 'meta.llama3-70b-instruct-v1')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in get_model_ids():
    print(f'错误：模型ID {model_id} 不是有效的模型ID。请将 MODEL_ID 环境变量设置为以下之一: {get_model_ids()}')
    sys.exit(0)

api_request = api_request_list[model_id]
config = {
    'log_level': 'none',
    'region': aws_region,
    'bedrock': {
        'api_request': api_request
    }
}

bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])

def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)

class BedrockWrapper:
    def __init__(self):
        self.init_prompt = [
            {"role": "system", "content": "你是一位人民教师，你的工作是给学生讲解各个科目的题目。"}
        ]
        self.conversation_history = self.init_prompt

    @staticmethod
    def define_body(text, history):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        # 检测输入语言
        try:
            detected_language = detect(text)
            #print(f"检测到的语言: {detected_language}")
        except Exception as e:
            print(f"语言检测失败: {e}")
            detected_language = 'zh-cn'  # 默认中文
        
        # 根据检测到的语言设置输出语言
        output_language = 'Chinese' if detected_language == 'zh-cn' else 'English'
        #print(f"设置的输出语言: {output_language}")

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                conversation = "\n".join([
                    f"<|start_header_id|>{msg['role']}<|end_header_id|>{msg['content']}<|eot_id|>"
                    for msg in history
                ])
                body['prompt'] = f"""
                    <|begin_of_text|>
                    {conversation}
                    <|start_header_id|>user<|end_header_id|>
                    {text}, please output in {output_language}.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else:
                conversation = "\n".join([
                    f"[INST] {msg['content']} [/INST]" if msg['role'] == 'user' 
                    else msg['content']
                    for msg in history
                ])
                body['prompt'] = f"<s>{conversation}[INST] {text}, please output in {output_language}. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                body['messages'] = history + [{"role": "user", "content": text}]
            else:
                conversation = "\n\n".join([
                    f"Human: {msg['content']}\n\nAssistant:" if msg['role'] == 'user'
                    else msg['content']
                    for msg in history
                ])
                body['prompt'] = f'{conversation}\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            conversation = "\n".join([
                f"User: {msg['content']}" if msg['role'] == 'user'
                else f"Assistant: {msg['content']}"
                for msg in history
            ])
            body['prompt'] = f"{conversation}\nUser: {text}"
        elif model_provider == 'mistral':
            conversation = "\n".join([
                f"[INST] {msg['content']} [/INST]" if msg['role'] == 'user'
                else msg['content']
                for msg in history
            ])
            body['prompt'] = f"<s>{conversation}[INST] {text}, please output in {output_language}. [/INST]"
        else:
            raise Exception('未知的模型提供商。')

        return body

    @staticmethod
    def get_response_text(chunk):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        
        chunk_obj = json.loads(chunk.get('bytes').decode())
        
        if model_provider == 'amazon':
            return chunk_obj['outputText']
        elif model_provider == 'meta':
            return chunk_obj['generation']
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                if chunk_obj['type'] == 'content_block_delta':
                    if chunk_obj['delta']['type'] == 'text_delta':
                        return chunk_obj['delta']['text']
                return ''
            return chunk_obj['completion']
        elif model_provider == 'cohere':
            return ' '.join([c["text"] for c in chunk_obj['generations']])
        elif model_provider == 'mistral':
            return chunk_obj['outputs'][0]['text']
        else:
            raise Exception('未知的模型提供商。')

    def chat(self, text):
        try:
            body = self.define_body(text, self.conversation_history)
            printer(f"[DEBUG] Request body: {body}", 'debug')
            
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=json.dumps(body),
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            full_response = ""
            for event in response.get('body'):
                text_chunk = self.get_response_text(event.get('chunk', {}))
                if text_chunk:
                    print(text_chunk, end='', flush=True)
                    full_response += text_chunk
            print('\n')

            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            # 保持历史记录在合理范围内（最近10轮对话）
            if len(self.conversation_history) > 20:
                self.conversation_history = self.init_prompt + self.conversation_history[-18:]
            
        except Exception as e:
            print(f"错误: {e}")

def process_image(image_name, text_request):
    # 构建图像路径
    image_path = os.path.join('.\\img', image_name)
    
    # 打开图像并进行 OCR
    try:
        img = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(img, lang='chi_sim')  # 使用中文 OCR
    except Exception as e:
        print(f"无法处理图像 {image_name}: {e}")
        return None
    
    # 将 OCR 文本与用户的文字需求结合
    combined_text = f"{ocr_text}\n\n{text_request}"
    return combined_text

def main():
    print(f'''
*************************************************************
[INFO] 支持的基础模型: {get_model_ids()}
[INFO] 通过设置 MODEL_ID 环境变量来更改模型。例如: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock 模型: {config['bedrock']['api_request']['modelId']}
[INFO] 日志级别: {config['log_level']}

[INFO] 现在可以直接输入文字进行对话！
[INFO] 输入 'quit' 退出程序
*************************************************************
''')

    bedrock = BedrockWrapper()
    
    while True:
        try:
            user_input = input("\n请输入您的问题或命令 (输入 'quit' 退出, 'clear' 清除历史): ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                bedrock.conversation_history = bedrock.init_prompt
                print("已清除对话历史")
                continue
            
            if user_input.startswith('img '):  # 检测 img 命令
                parts = user_input.split(' ', 2)
                if len(parts) < 3:
                    print("命令格式错误，应为: img 图片名称 文字需求")
                    continue
                image_name, text_request = parts[1], parts[2]
                print(f"正在处理图像 {image_name} 并生成文字需求: {text_request}")
                combined_text = process_image(image_name, text_request)
                if combined_text:
                    bedrock.chat(combined_text)
            else:
                bedrock.chat(user_input)
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"发生错误: {e}")


main()
