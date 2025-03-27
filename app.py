import json
import os
import sys
import boto3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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

# 定义 printer 函数
def printer(text, level):
    if config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)

bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])

# 加载《红楼梦》全文
def load_hongloumeng_text():
    file_path = r'C:\Users\31822\FreshmanCup\docs\honglou.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 分割文本为段落
def split_text_into_passages(text, passage_length=100):
    words = text.split()
    passages = [' '.join(words[i:i + passage_length]) for i in range(0, len(words), passage_length)]
    return passages

# 生成嵌入并构建FAISS索引
def build_faiss_index(passages):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    passage_embeddings = model.encode(passages)
    dimension = passage_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(passage_embeddings)
    return index, model, passages

# 检索最相关的段落
def retrieve_relevant_passages(query, index, model, passages, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_passages = [passages[i] for i in indices[0]]
    return relevant_passages

class BedrockWrapper:
    def __init__(self):
        self.conversation_history = []
        # 加载《红楼梦》全文并构建FAISS索引
        self.hongloumeng_text = load_hongloumeng_text()
        self.passages = split_text_into_passages(self.hongloumeng_text)
        self.index, self.embedding_model, self.passages = build_faiss_index(self.passages)

    def define_body(self, text, context=None):
        model_id = config['bedrock']['api_request']['modelId']
        model_provider = model_id.split('.')[0]
        body = config['bedrock']['api_request']['body']

        # 将对话历史和检索到的上下文添加到当前输入
        if self.conversation_history:
            text = "\n".join(self.conversation_history) + "\n" + text
        if context:
            text = f"上下文:\n{context}\n\n问题:\n{text}"

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
            # 检索最相关的段落
            relevant_passages = retrieve_relevant_passages(text, self.index, self.embedding_model, self.passages)
            context = "\n".join(relevant_passages)

            # 生成回答
            body = self.define_body(text, context)
            printer(f"[DEBUG] Request body: {body}", 'debug')
            
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=json.dumps(body),
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            response_text = ''
            for event in response.get('body'):
                text_chunk = self.get_response_text(event.get('chunk', {}))
                if text_chunk:
                    response_text += text_chunk
                    print(text_chunk, end='', flush=True)
            print('\n')

            # 将用户输入和模型响应添加到对话历史中
            self.conversation_history.append(f"用户: {text}")
            self.conversation_history.append(f"助手: {response_text}")

            # 限制对话历史长度，避免内存溢出
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
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