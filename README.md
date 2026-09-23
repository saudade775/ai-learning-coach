# ai-learning-coach
AI 学习陪跑顾问

一个懂你的 AI 学习陪跑系统。它会记住你的学习情况，陪着你一步步学 AI。

## 功能

- 首次使用填写学习档案
- 支持上传 PDF / TXT / PPTX / DOCX 学习资料
- 基于资料回答问题
- 自动提取记忆，越用越懂你
- 体验模式 10 次免费，也可绑定自己的 API Key

## 技术栈

- Python
- Streamlit
- LangChain
- Chroma
- 智谱AI

## 如何运行

1. 安装依赖：
   pip install -r requirements.txt

2. 设置环境变量：
   ZHIPU_API_KEY=你的Key

3. 运行：
   streamlit run app.py
