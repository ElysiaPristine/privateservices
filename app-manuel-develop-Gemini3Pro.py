# app-manuel-stable-release.py - 集成文档上传功能的完整应用（Gemini版本）
import os
import json
import chainlit as cl
from typing import List, Dict, Any
import tempfile
import asyncio
import datetime
import google.generativeai as genai
import httpx
from PyPDF2 import PdfReader

# 配置 Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "你的Gemini API密钥")
# 如果需要代理，可以在这里配置
PROXY_URL = os.getenv("example_ip&port")  # 例如 "http://127.0.0.1:8080"

# 配置 Gemini
if PROXY_URL:
    http_client = httpx.Client(proxies=PROXY_URL)
    genai.configure(
        api_key=GEMINI_API_KEY,
        http_client=http_client
    )
else:
    genai.configure(api_key=GEMINI_API_KEY)


class ChatHistoryManager:
    """聊天历史记录管理器"""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str):
        """添加消息到历史记录"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.history.copy()

    def clear_history(self):
        """清空历史记录"""
        self.history.clear()

    def export_history(self) -> str:
        """导出历史记录为JSON字符串"""
        return json.dumps({
            "export_time": datetime.datetime.now().isoformat(),
            "history": self.history
        }, ensure_ascii=False, indent=2)

    def import_history(self, json_data: str) -> bool:
        """从JSON字符串导入历史记录"""
        try:
            data = json.loads(json_data)
            if "history" in data and isinstance(data["history"], list):
                self.history = data["history"]
                return True
        except Exception as e:
            print(f"导入历史记录失败: {e}")
        return False


class DocumentProcessor:
    """文档处理器"""

    @staticmethod
    async def extract_text_from_file(file_path: str, file_type: str) -> str:
        """
        从文件中提取文本内容

        Args:
            file_path: 文件路径
            file_type: 文件类型

        Returns:
            提取的文本内容
        """
        try:
            if file_type == "text/plain":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif file_type == "application/pdf":
                return await DocumentProcessor._extract_from_pdf(file_path)

            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return await DocumentProcessor._extract_from_docx(file_path)

            else:
                return f"不支持的文件类型: {file_type}"

        except Exception as e:
            return f"文件处理失败: {str(e)}"

    @staticmethod
    async def _extract_from_pdf(file_path: str) -> str:
        """从PDF提取文本"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"PDF处理失败: {str(e)}"

    @staticmethod
    async def _extract_from_docx(file_path: str) -> str:
        """从DOCX提取文本"""
        try:
            # 使用 python-docx2txt 替代 python-docx
            import docx2txt
            text = docx2txt.process(file_path)
            return text
        except ImportError:
            # 如果 docx2txt 不可用，尝试使用其他方法
            try:
                # 使用 zipfile 直接解析 docx 文件
                import zipfile
                import xml.etree.ElementTree as ET

                # docx 文件实际上是 zip 包
                with zipfile.ZipFile(file_path) as docx:
                    # 读取文档内容
                    document_xml = docx.read('word/document.xml')

                    # 解析 XML
                    root = ET.fromstring(document_xml)

                    # 提取所有文本
                    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    text_elements = root.findall('.//w:t', namespaces)

                    text = ""
                    for elem in text_elements:
                        if elem.text:
                            text += elem.text + " "

                    return text.strip()
            except Exception as e:
                return f"DOCX处理失败: 无法解析文档 - {str(e)}"
        except Exception as e:
            return f"DOCX处理失败: {str(e)}"


# 初始化文档处理器
doc_processor = DocumentProcessor()


@cl.on_chat_start
async def start_chat():
    """聊天开始时的初始化"""
    # 初始化聊天历史管理器
    history_manager = ChatHistoryManager()
    cl.user_session.set("history_manager", history_manager)

    # 初始化文档内容存储
    cl.user_session.set("document_content", "")

    welcome_msg = """你好！我是你的个人AI助手（基于Gemini API），支持文档上传和解析功能。

你可以使用以下命令：
- `/upload` - 上传文档（支持txt、pdf、docx格式）
- `/export` - 导出当前聊天记录
- `/import` - 导入聊天记录文件
- `/clear` - 清空当前聊天记录
- `/summarize` - 总结已上传的文档内容

你也可以直接输入文本进行普通聊天。"""

    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """处理消息"""
    user_message = message.content

    # 获取历史管理器
    history_manager = cl.user_session.get("history_manager")
    if history_manager is None:
        history_manager = ChatHistoryManager()
        cl.user_session.set("history_manager", history_manager)

    # 检查命令
    if user_message == '/upload':
        await handle_document_upload(message, history_manager)
        return
    elif user_message == '/export':
        await export_chat_history(message, history_manager)
        return
    elif user_message == '/import':
        await import_chat_history(message, history_manager)
        return
    elif user_message == '/clear':
        await clear_chat_history(message, history_manager)
        return
    elif user_message == '/summarize':
        await summarize_document(message, history_manager)
        return

    # 普通聊天消息处理
    await handle_chat_message(user_message, history_manager)


async def handle_chat_message(user_message: str, history_manager: ChatHistoryManager):
    """处理普通聊天消息（使用Gemini API）"""
    # 添加用户消息到历史
    history_manager.add_message("user", user_message)

    try:
        # 初始化Gemini模型
        model = genai.GenerativeModel('gemini-pro')

        # 构建对话历史
        chat_history = []
        for msg in history_manager.get_history():
            if msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            else:
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        # 开始聊天会话
        chat = model.start_chat(history=chat_history[:-1])  # 排除当前用户消息

        # 发送当前消息并获取响应
        response = chat.send_message(user_message, stream=True)

        msg = cl.Message(content="")
        assistant_response = ""

        # 流式响应
        for chunk in response:
            if chunk.text:
                assistant_response += chunk.text
                await msg.stream_token(chunk.text)

        # 添加助手回复到历史
        history_manager.add_message("assistant", assistant_response)
        await msg.update()

    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        await cl.Message(content=error_msg).send()


async def handle_document_upload(message: cl.Message, history_manager: ChatHistoryManager):
    """处理文档上传"""
    # 等待用户上传文件
    files = await cl.AskFileMessage(
        content="请上传文档文件（支持txt、pdf、docx格式）",
        accept=[
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        max_size_mb=10,
        timeout=300,
    ).send()

    if files:
        file = files[0]
        await message.send(f"正在处理文档: {file.name}")

        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                # 写入文件内容
                with open(file.path, 'rb') as f:
                    tmp.write(f.read())
                tmp_path = tmp.name

            # 提取文本内容
            extracted_text = await doc_processor.extract_text_from_file(tmp_path, file.type)

            # 保存文档内容到会话
            cl.user_session.set("document_content", extracted_text)

            # 显示文档预览（前500字符）
            preview = extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")

            # 添加系统消息到历史
            system_msg = f"用户上传了文档 '{file.name}'，内容已提取并可用于分析。"
            history_manager.add_message("system", system_msg)

            # 发送处理结果
            elements = [
                cl.File(name=file.name, path=file.path, display="inline"),
            ]

            await cl.Message(
                content=f"文档 '{file.name}' 处理完成！\n\n**文档预览:**\n{preview}\n\n你可以使用 `/summarize` 命令总结文档内容，或直接询问关于文档的问题。",
                elements=elements
            ).send()

            # 清理临时文件
            os.unlink(tmp_path)

        except Exception as e:
            await cl.Message(content=f"文档处理失败: {str(e)}").send()


async def summarize_document(message: cl.Message, history_manager: ChatHistoryManager):
    """总结文档内容"""
    document_content = cl.user_session.get("document_content", "")

    if not document_content:
        await message.send("请先使用 `/upload` 命令上传文档")
        return

    await message.send("正在总结文档内容...")

    try:
        # 初始化Gemini模型
        model = genai.GenerativeModel('gemini-pro')

        # 构建总结提示
        prompt = f"请总结以下文档内容，提取关键信息并以简洁明了的方式呈现：\n\n{document_content}"

        # 发送总结请求
        response = model.generate_content(prompt)

        # 添加总结到历史
        history_manager.add_message("assistant", f"文档总结：\n{response.text}")

        await cl.Message(content=f"**文档总结：**\n\n{response.text}").send()

    except Exception as e:
        await cl.Message(content=f"文档总结失败: {str(e)}").send()


async def export_chat_history(message: cl.Message, history_manager: ChatHistoryManager):
    """导出聊天记录"""
    if not history_manager.get_history():
        await message.send("当前没有聊天记录可导出")
        return

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
            json_data = history_manager.export_history()
            tmp.write(json_data)
            tmp_path = tmp.name

        # 发送文件
        elements = [
            cl.File(name="chat_history.json", path=tmp_path, display="inline"),
        ]
        await cl.Message(content="聊天记录导出成功", elements=elements).send()

        # 清理临时文件
        os.unlink(tmp_path)
    except Exception as e:
        await cl.Message(content=f"导出失败: {str(e)}").send()


async def import_chat_history(message: cl.Message, history_manager: ChatHistoryManager):
    """导入聊天记录"""
    # 等待用户上传文件
    files_response = await cl.AskFileMessage(
        content="请上传之前导出的聊天记录JSON文件",
        accept=["application/json"],
        max_size_mb=10,
        timeout=300,
    ).send()

    if files_response:
        try:
            # 读取文件内容
            file = files_response[0]
            with open(file.path, 'r', encoding='utf-8') as f:
                import_content = f.read()

            # 导入历史记录
            if history_manager.import_history(import_content):
                await cl.Message(content=f"成功导入聊天记录，共 {len(history_manager.get_history())} 条消息").send()
            else:
                await cl.Message(content="导入失败：文件格式不正确").send()
        except Exception as e:
            await cl.Message(content=f"导入失败: {str(e)}").send()


async def clear_chat_history(message: cl.Message, history_manager: ChatHistoryManager):
    """清空聊天记录"""
    history_manager.clear_history()
    # 同时清空文档内容
    cl.user_session.set("document_content", "")
    await cl.Message(content="聊天记录和文档内容已清空").send()


if __name__ == "__main__":
    print("请使用 'chainlit run app-manuel-develop-Gemini3Pro.py' 命令启动应用")