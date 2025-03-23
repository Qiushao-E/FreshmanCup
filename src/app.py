import json
import os
import sys
import boto3
from api_request_schema import api_request_list, get_model_ids

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
    @staticmethod
    def define_body(text):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        if model_provider == 'amazon':
            body['inputText'] = text
        elif model_provider == 'meta':
            if 'llama3' in model_id:
                body['prompt'] = f"""
                    <|begin_of_text|>
                    <|start_header_id|>user<|end_header_id|>
                    {text}, please output in Chinese.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """
            else: 
                body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
        elif model_provider == 'anthropic':
            if "claude-3" in model_id:
                body['messages'] = [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            else:
                body['prompt'] = f'\n\nHuman: {text}\n\nAssistant:'
        elif model_provider == 'cohere':
            body['prompt'] = text
        elif model_provider == 'mistral':
            body['prompt'] = f"<s>[INST] {text}, please output in Chinese. [/INST]"
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
            body = self.define_body(text)
            printer(f"[DEBUG] Request body: {body}", 'debug')
            
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=json.dumps(body),
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            for event in response.get('body'):
                text = self.get_response_text(event.get('chunk', {}))
                if text:
                    print(text, end='', flush=True)
            print('\n')
            
        except Exception as e:
            print(f"错误: {e}")

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
            user_input = input("\n请输入您的问题 (输入 'quit' 退出): ")
            if user_input.lower() == 'quit':
                break
            bedrock.chat(user_input)
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"发生错误: {e}")


main()
