import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
import base64
PICTURE_NAME='CV_1.pdf'





async def main(): 
    # 1. initialize
    client = MultiServerMCPClient({
    "social_graph": {
        "transport": "http",
        "url": "https://ftec5660.ngrok.app/mcp",
        "headers": {"ngrok-skip-browser-warning": "true"}
    }
    })


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv("v_api"),
        vertexai=True,
        temperature=0.2,
    )
    # 2. load tools from MCP server
    print("Fetching tools from MCP server...")
    try:
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error fetching tools: {e}")
        return

    # 3. bind tools to LLM
    llm_with_tools = llm.bind_tools(mcp_tools)

    # 4.read CV
    prompt_read='Describe these CVs in detail.And identify the candidates from the CVs.'
    
    cv_dir = os.path.join(os.path.dirname(__file__), 'cv')
    
    if not os.path.exists(cv_dir):
        print(f"Error: Directory not found at {cv_dir}")
        return


    message_content = [{"type": "text", "text": prompt_read}]
    

    pdf_files = [f for f in os.listdir(cv_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in 'cv' directory.")
        return
        
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    for pdf_file in pdf_files:
        picture_path = os.path.join(cv_dir, pdf_file)
        
        try:
            with open(picture_path, "rb") as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode("utf-8")

            mime_type = "application/pdf"

            media_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}
            }
            
            message_content.append(media_content)
            
        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")

    message = HumanMessage(content=message_content)

    # 5. invoke LLM with tools to get the information from CV
    response = llm_with_tools.invoke([message]).content
    print(f'CV information: \n{response}')

    prompt_facebook=f'''
    This is the information from the CVs:
    {response}
    Please identify the candidates from the CVs.
    For each candidate found, search for their information on LinkedIn/Facebook to get their profile.
    Detect discrepancies between CV claims and social media data for each candidate.
    Generate a verification report for each candidate.
    You need give each of the candidates a score between 0 and 1 based on the verification report.
    Each score must be a float in the range [0, 1], representing the reliability or confidence that the CV is valid (or meets the task criteria).
    Each CV is evaluated independently using a threshold of 0.5:
    If score > 0.5 and groundtruth == 1 → Full credit
    If score ≤ 0.5 and groundtruth == 0 → Full credit
    Otherwise → No credit
    '''
    
    # 6. build the tool map
    tool_map = {tool.name: tool for tool in mcp_tools}

    messages = [HumanMessage(content=prompt_facebook)]
    
    print("\n--- Starting Facebook Verification Process ---")

    for i in range(20):
        print(f"Iteration {i+1}: Invoking LLM...")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # 1. if there are tool calls, execute the tools
        if ai_msg.tool_calls:
            print(f"Tool calls detected: {len(ai_msg.tool_calls)}")
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                print(f"  -> Calling tool: {tool_name} with args: {tool_args}")
                
                try:
                    selected_tool = tool_map.get(tool_name)
                    if selected_tool:
                        # 2. if the tool is async, use ainvoke
                        tool_output = await selected_tool.ainvoke(tool_args)
                    else:
                        tool_output = f"Error: Tool {tool_name} not found."
                        
                    print(f"  <- Tool output: {str(tool_output)[:100]}...")
                except Exception as e:
                    tool_output = f"Error executing tool {tool_name}: {e}"
                    print(f"  <- Tool execution error: {e}")
                
                # 3. add the tool output as a ToolMessage to the messages
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id, name=tool_name))
        
        # 2. if there are no tool calls, the LLM has generated the final reply, print and exit
        else:
            print(f'\nFacebook information: \n{ai_msg.content}')
            break



if __name__ == "__main__":
    asyncio.run(main())

    