from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated, Literal

import os
from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0.2)
    print(llm.invoke("What is a capital of Thailand?"))
except:
    ValueError (print("Op! something wrong about your model"))

class State(TypedDict):
    messages : Annotated[list[str],add_messages]
    decision : str
    text_edit : str

def content_creator(state: State) -> State:

    print("--- Content Creator Agent ---")
    ''' Content creator agent for education'''
    system = """You are content creator agent, Your task is create high quality content from title that user given.
                This content must be concise and easy to understand for 15 year old. this content will be used for educational purpose.
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",system),
        ("human","{messages}")
    ])
    
    chain = prompt | llm 

    result = chain.invoke({'messages': state["messages"]})

    return {'messages' : result.content}

def refine_creator(state: State) -> State:

    print("--- Content Creator Agent ---")
    ''' Refine content creator agent for publish. Purpose for educational.'''
    system = """You are content creator agent, Your task is analysis content given and rewrite it related to user requiremetns
                Purpose for education.
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",system),
        MessagesPlaceholder(variable_name="old_content"),
        ("human","{decision}")
    ])

    old_content = state["messages"][-1:]
    
    chain = prompt | llm 

    result = chain.invoke({'decision': state["messages"], "old_content": old_content})

    return {'messages' : result.content, 'publish': result.content}

def human_review(state : State) -> Command[Literal[END, "refine_creator"]]:
    ''' Human approve the content '''
    
    values = interrupt({
                        "question":'''Is this pass your requirement?''',
                        "content": state['messages'][-1],
                        # "decision": state['decision']
                        })
    if values == "approved":
        print("--- Human Approve process ---")
        return Command(goto=END, update={"decision": "approved"})
    else:
        print("--- Rewrite process ---")
        return Command(goto="refine_creator", update={"decision": values})
    
checkpointer = MemorySaver()

graph = StateGraph(State)
graph.add_node("content_creator", content_creator)
#####
graph.add_node("refine_creator", refine_creator)
graph.add_node("human",human_review)
#####
graph.add_edge(START, "content_creator")
graph.add_edge("content_creator", "human")
graph.add_edge("human",END)
workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": 4}}

    text = input("Enter your content request: ")
    
    r = workflow.invoke({"messages":text}, config=config)

    print(r)
    
    text = input("Enter your feedback (or type 'approved' to approve): ")

    if text == "approved":
        r = workflow.invoke(Command(resume=text),config=config)
    else:
        r = workflow.invoke(Command(resume={"decision": text, 
                                            "old_content": r["__interrupt__"][0].value["content"].content}),config=config)
    print(r)