import os
import base64
from PIL import Image
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field


# 图像预处理
def process_image(image_path, max_size=512):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


#输出格式
class receipt(BaseModel):
    commodity: dict[str,list] = Field(description="商品名称:[数量，价格,编号....]")
    summary: dict[str,float] = Field(description="除去商品本身的统计信息，例如总价，折扣，支付金额等")
    
    

if __name__ == '__main__':
    folder = r"H:\desk\python\ai_agent\HWagent\pictures" + "\\"
    picture_name = "1.jpg"
    picture_path = folder + picture_name
    image_base64 = process_image(picture_path)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("v_api"),
        vertexai=True,
        temperature=0.2,
    )

    message=HumanMessage(
        content=[
            {"type": "text", "text": '''
            你是一个专业的图像描述员，你需要提取图片中购物小票的内容，包括商品名称、数量、单价、金额、总金额等信息。
            '''},
            {
            "type": "image",
            "base64": image_base64,
            "mime_type": "image/jpeg",
            },
        ],
    )

    llm_structured = llm.with_structured_output(receipt)
    result = llm_structured.invoke([message])
    print(type(result),result)

    # 构建问答Chain
    prompt_template = ChatPromptTemplate.from_template(
        """你是一个智能助手，请根据以下购物小票的结构化数据回答用户的问题。
        
        小票数据:
        {receipt_data}
        
        用户问题: {question}
        """
    )

    chain = prompt_template | llm | StrOutputParser()

    # 模拟用户提问
    user_question = "帮我算一下这单总共花了多少钱？最贵的东西是什么？"
    print(f"\n用户提问: {user_question}")
    
    answer = chain.invoke({"receipt_data": str(result), "question": user_question})
    print(f"AI回答: {answer}")






    
   
