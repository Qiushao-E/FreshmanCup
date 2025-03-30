import json
import os
import sys
import boto3
from api_request_schema import api_request_list, get_model_ids
import pytesseract
from PIL import Image
from langdetect import detect
from prompt_template import generate_teaching_prompt
import speech_recognition as sr
import pyaudio
# 定语从句教学示例
grammar_prompt = generate_teaching_prompt(
    topic="定语从句",
    core_elements="基本概念、结构和用法",
    learning_goals="实际应用能力",
    structure_elements="概念讲解、句型分析、实例讲解",
    teaching_tools="例句分析",
    thinking_direction="日常语言中的应用"
)
    
# 第一次世界大战教学示例
history_prompt = generate_teaching_prompt(
    topic="第一次世界大战",
    core_elements="背景、起因、过程和影响",
    learning_goals="历史思维能力",
    structure_elements="背景介绍、战争过程、影响分析",
    teaching_method="启发引导",
    teaching_tools="历史案例",
    thinking_direction="对当今世界的影响"
)


# 数学教学示例
math_prompt = generate_teaching_prompt(
    topic="二次函数",
    core_elements="定义、图像、性质",
    learning_goals="应用能力",
    structure_elements="定义、图像、性质",
    teaching_method="启发引导",
    teaching_tools="分析例题",
    thinking_direction="实际应用"
)

# 数据结构
data_structure_prompt = generate_teaching_prompt(
    topic="数据结构",
    core_elements="基本概念、常见数据结构类型、时间复杂度分析",
    learning_goals="数据结构选择与应用能力",
    structure_elements="概念讲解、结构分析、性能对比、应用场景",
    content_requirements="重点突出实际应用",
    teaching_method="案例分析",
    teaching_tools="代码示例和性能分析",
    thinking_direction="如何选择合适的数据结构解决实际问题",
    interaction_type="编程实践",
    interaction_goal="培养算法思维",
    word_count="不少于1000字",
    language_style="通俗易懂",
    special_requirements="结合具体编程语言和实际项目案例"
)


# 默认提示词
default_prompt = """你是一位经验丰富的人民教师，有着20年的教学经验。你的职责是：
1. 耐心细致地为学生讲解各个学科（数学、编程、数据结构等）的课程内容和习题
2. 采用循序渐进的教学方式，先确保学生理解基础概念，再逐步深入
3. 解答问题时要：
   - 先分析题目要点
   - 给出清晰的解题思路
   - 详细说明每一步的原理
   - 适时补充相关知识点
4. 鼓励学生思考，引导而不是直接给出答案
5. 使用亲切友好的语气，营造良好的师生互动氛围
6. 当遇到学生不理解的地方，要用更通俗易懂的方式重新解释

记住：你的目标不仅是解答问题，更要培养学生的学习兴趣和独立思考能力。"""

prompt = data_structure_prompt

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
            {"role": "system", "content": prompt}
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
            print(f"检测到的语言: {detected_language}")
        except Exception as e:
            print(f"语言检测失败: {e}")
            detected_language = 'zh-cn'  # 默认中文
        
        # 根据检测到的语言设置输出语言
        output_language = 'English' if detected_language == 'en' else 'Chinese'
        print(f"设置的输出语言: {output_language}")

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

def record_audio():
    """录制音频并转换为文字"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("请开始说话...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("正在识别...")
            
            # 首先尝试中文识别
            try:
                text = r.recognize_google(audio, language='zh-CN')
                return text
            except:
                # 如果中文识别失败，尝试英文识别
                try:
                    text = r.recognize_google(audio, language='en-US')
                    return text
                except sr.UnknownValueError:
                    print("无法识别您的语音")
                    return None
                except sr.RequestError as e:
                    print(f"无法连接到 Google 语音识别服务：{e}")
                    return None
        except sr.WaitTimeoutError:
            print("没有检测到语音输入")
            return None
        except Exception as e:
            print(f"发生错误：{e}")
            return None

def get_multiline_input():
    """获取多行输入，直到用户输入空行为止"""
    print("请输入内容（输入空行结束）：")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    print(f'''
*************************************************************
[INFO] 支持的基础模型: {get_model_ids()}
[INFO] 通过设置 MODEL_ID 环境变量来更改模型。例如: export MODEL_ID=meta.llama2-70b-chat-v1

[INFO] AWS Region: {config['region']}
[INFO] Amazon Bedrock 模型: {config['bedrock']['api_request']['modelId']}
[INFO] 日志级别: {config['log_level']}

[INFO] 使用说明：
- 输入 'quit' 退出程序
- 输入 'clear' 清除对话历史
- 输入 'speak' 使用语音输入
- 输入 'multi' 进入多行输入模式
- 输入 'img 图片名称 文字需求' 处理图片
*************************************************************
''')

    bedrock = BedrockWrapper()
    
    while True:
        try:
            user_input = input("\n请输入您的问题或命令: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                bedrock.conversation_history = [
                    {"role": "system", "content": prompt}
                ]
                print("已清除对话历史")
                continue
            elif user_input.lower() == 'speak':
                text = record_audio()
                if text:
                    print(f"识别到的文字: {text}")
                    bedrock.chat(text)
                continue
            elif user_input.lower() == 'multi':
                text = get_multiline_input()
                if text.strip():
                    bedrock.chat(text)
                continue
            
            if user_input.startswith('img '):
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
