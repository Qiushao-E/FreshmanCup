TEACHING_PROMPT_TEMPLATE = """
你是一位经验丰富的人民教师，有着20年的教学经验。

现在，你需要向学生讲解"{主题}"。

具体要求如下：
1. 教学目标：
   - 帮助学生理解{主题}的{核心要素}
   - 培养学生对{主题}的{学习目标}

2. 内容结构：
   - 需要按照{结构要素}等逻辑顺序进行讲解
   - 每个部分要{内容要求}
   - 确保内容的连贯性和层次性

3. 教学方法：
   - 采用{教学方式}的方式
   - 通过{教学手段}帮助学生理解
   - 鼓励学生思考{思考方向}

4. 互动设计：
   - 在适当位置设置{互动形式}
   - 引导学生进行{互动目标}

5. 输出要求：
   - 字数要求：{具体字数要求}
   - 语言风格：{语言要求}
   - 其他特殊要求：{如有其他具体要求}
"""

def generate_teaching_prompt(
    topic: str,
    core_elements: str,
    learning_goals: str,
    structure_elements: str,
    content_requirements: str = "简明扼要",
    teaching_method: str = "逐步思考",
    teaching_tools: str = "实例分析",
    thinking_direction: str = "实际应用",
    interaction_type: str = "思考问题",
    interaction_goal: str = "知识应用",
    word_count: str = "不少于1000字",
    language_style: str = "简明易懂",
    special_requirements: str = ""
) -> str:
    """
    生成教学提示词
    
    参数:
        topic: 教学主题
        core_elements: 核心要素
        learning_goals: 学习目标
        structure_elements: 结构要素
        content_requirements: 内容要求
        teaching_method: 教学方式
        teaching_tools: 教学手段
        thinking_direction: 思考方向
        interaction_type: 互动形式
        interaction_goal: 互动目标
        word_count: 字数要求
        language_style: 语言风格
        special_requirements: 特殊要求
    
    返回:
        str: 生成的提示词
    """
    return TEACHING_PROMPT_TEMPLATE.format(
        主题=topic,
        核心要素=core_elements,
        学习目标=learning_goals,
        结构要素=structure_elements,
        内容要求=content_requirements,
        教学方式=teaching_method,
        教学手段=teaching_tools,
        思考方向=thinking_direction,
        互动形式=interaction_type,
        互动目标=interaction_goal,
        具体字数要求=word_count,
        语言要求=language_style,
        如有其他具体要求=special_requirements
    )

# 使用示例：
if __name__ == "__main__":
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
    
    print("语法教学提示词示例：")
    print(grammar_prompt)
    print("\n历史教学提示词示例：")
    print(history_prompt) 