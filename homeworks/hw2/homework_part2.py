import os
import requests
import json
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# Configuration
API_KEY = "moltbook_sk_ozh1HJwkxExe_VG_Fmoq2Sgt5m1upXyE"
BASE_URL = "https://www.moltbook.com/api/v1"
AGENT_NAME = "nickname_1596"

# Helper for API requests
def _make_request(method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data)
        else:
            return {"error": f"Unsupported method: {method}"}
            
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"status": response.status_code, "text": response.text}
            
    except Exception as e:
        return {"error": str(e)}

# --- Tools ---

@tool
def get_feed(sort: str = "hot", limit: int = 10, filter_type: str = "all") -> str:
    """
    Get the latest posts from the Moltbook feed.
    
    Args:
        sort: Sort order. Options: 'hot', 'new', 'top', 'rising'. Default is 'hot'.
        limit: Number of posts to return. Default is 10.
        filter_type: Filter by 'all' (subscriptions + follows) or 'following' (only follows).
    """
    endpoint = "/feed"
    params = {"sort": sort, "limit": limit, "filter": filter_type}
    result = _make_request("GET", endpoint, params=params)
    return json.dumps(result, indent=2)

@tool
def subscribe_to_submolt(submolt_name: str) -> str:
    """
    Subscribe to a submolt (community).
    
    Args:
        submolt_name: The name of the submolt to subscribe to (e.g., 'aithoughts').
    """
    endpoint = f"/submolts/{submolt_name}/subscribe"
    result = _make_request("POST", endpoint)
    return json.dumps(result, indent=2)

@tool
def upvote_post(post_id: str) -> str:
    """
    Upvote a specific post.
    
    Args:
        post_id: The ID of the post to upvote.
    """
    endpoint = f"/posts/{post_id}/upvote"
    result = _make_request("POST", endpoint)
    return json.dumps(result, indent=2)

@tool
def upvote_comment(comment_id: str) -> str:
    """
    Upvote a specific comment.
    
    Args:
        comment_id: The ID of the comment to upvote.
    """
    endpoint = f"/comments/{comment_id}/upvote"
    result = _make_request("POST", endpoint)
    return json.dumps(result, indent=2)

@tool
def add_comment(post_id: str, content: str, parent_id: Optional[str] = None) -> str:
    """
    Add a comment to a post or reply to another comment.
    
    Args:
        post_id: The ID of the post to comment on.
        content: The text content of the comment.
        parent_id: Optional. The ID of the comment to reply to.
    """
    endpoint = f"/posts/{post_id}/comments"
    data = {"content": content}
    if parent_id:
        data["parent_id"] = parent_id
        
    result = _make_request("POST", endpoint, data=data)
    
    # Check for verification challenge (simple handling)
    if result.get("verification_required"):
        return f"Verification required. Challenge: {result.get('verification', {}).get('challenge_text')}. (Automatic solving not implemented yet)"
        
    return json.dumps(result, indent=2)

@tool
def get_submolts() -> str:
    """List all available submolts."""
    endpoint = "/submolts"
    result = _make_request("GET", endpoint)
    return json.dumps(result, indent=2)

@tool
def search_content(query: str, type: str = "all", limit: int = 10) -> str:
    """
    Search for posts and comments using semantic search.
    
    Args:
        query: The search query (natural language).
        type: 'posts', 'comments', or 'all'.
        limit: Max results.
    """
    endpoint = "/search"
    params = {"q": query, "type": type, "limit": limit}
    result = _make_request("GET", endpoint, params=params)
    return json.dumps(result, indent=2)

# Main Agent Logic
def main():
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=os.getenv("v_api"),
        vertexai=True,
        temperature=0.7,
    )
    
    # List of tools
    tools = [get_feed, subscribe_to_submolt, upvote_post, upvote_comment, add_comment, get_submolts, search_content]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Initial system message
    messages = [
        HumanMessage(content=f"You are an AI agent named {AGENT_NAME} on Moltbook. "
                             "You can browse the feed, subscribe to communities, upvote content, and leave comments. "
                             "Help the user interact with Moltbook based on their requests. "
                             "If you need to find something to upvote or comment on, check the feed or search first.")
    ]
    
    print(f"Agent {AGENT_NAME} initialized. Ready to chat! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages.append(HumanMessage(content=user_input))
        
        # Agent loop (ReAct-style)
        while True:
            print("Thinking...")
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            
            if ai_msg.tool_calls:
                print(f"Executing {len(ai_msg.tool_calls)} tool(s)...")
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    print(f"  -> Calling {tool_name} with {tool_args}")
                    
                    # Find and execute the tool
                    selected_tool = next((t for t in tools if t.name == tool_name), None)
                    if selected_tool:
                        try:
                            tool_output = selected_tool.invoke(tool_args)
                        except Exception as e:
                            tool_output = f"Error: {e}"
                    else:
                        tool_output = f"Error: Tool {tool_name} not found"
                        
                    print(f"  <- Result: {str(tool_output)[:100]}...") # Truncate log
                    
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id, name=tool_name))
            else:
                # Final response from AI
                print(f"\nAgent: {ai_msg.content}")
                break

if __name__ == "__main__":
    main()
