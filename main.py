import streamlit as st
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
import uuid

# Load environment variables
load_dotenv()

# Initialize LLM
@st.cache_resource
def get_llm():
    try:
        return ChatGoogleGenerativeAI(model='gemini-2.5-pro', temperature=0.2)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# State definition - exactly like in notebook
class State(TypedDict):
    messages: Annotated[list[str], add_messages]
    decision: str
    text_edit: str

# Agent functions - exactly like in notebook
def content_creator(state: State) -> State:
    st.write("--- Content Creator Agent ---")
    
    system = """You are content creator agent, Your task is create high quality content from title that user given.
                This content must be concise and easy to understand for 15 year old. this content will be used for educational purpose.
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{messages}")
    ])
    
    llm = get_llm()
    if not llm:
        return {'messages': 'Error: Could not initialize LLM'}
    
    chain = prompt | llm 
    result = chain.invoke({'messages': state["messages"]})
    
    return {'messages': result.content}

def refine_creator(state: State) -> State:
    st.write("--- Content Refiner Agent ---")
    
    system = """You are content creator agent, Your task is analysis content given and rewrite it related to user requirements
                Purpose for education.
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="old_content"),
        ("human", "{decision}")
    ])

    old_content = state["messages"][-1:]
    llm = get_llm()
    if not llm:
        return {'messages': 'Error: Could not initialize LLM'}
    
    chain = prompt | llm 
    result = chain.invoke({'decision': state["messages"], "old_content": old_content})

    return {'messages': result.content, 'publish': result.content}

# Human review function - exactly like in notebook with interrupt()
def human_review(state: State) -> Command[Literal[END, "refine_creator"]]:
    """Human approve the content"""
    
    values = interrupt({
        "question": '''Is this pass your requirement?''',
        "content": state['messages'][-1],
    })
    
    if values == "approved":
        st.write("--- Human Approve process ---")
        return Command(goto=END, update={"decision": "approved"})
    else:
        st.write("--- Rewrite process ---")
        return Command(goto="refine_creator", update={"decision": values})

# Initialize workflow - exactly like in notebook
@st.cache_resource
def create_workflow():
    checkpointer = MemorySaver()
    
    graph = StateGraph(State)
    graph.add_node("content_creator", content_creator)
    graph.add_node("refine_creator", refine_creator)
    graph.add_node("human", human_review)
    
    graph.add_edge(START, "content_creator")
    graph.add_edge("content_creator", "human")
    graph.add_edge("human", END)
    
    return graph.compile(checkpointer=checkpointer)

# Helper function to extract and store content
def extract_and_store_content(result, version_type="Original"):
    """Extract content from workflow result and store in history"""
    if '__interrupt__' in result:
        interrupt_data = result['__interrupt__'][0].value
        content = interrupt_data['content']
        if hasattr(content, 'content'):
            content_text = content.content
        else:
            content_text = str(content)
    else:
        # For completed workflows
        final_messages = result.get('messages', [])
        if final_messages:
            final_content = final_messages[-1]
            if hasattr(final_content, 'content'):
                content_text = final_content.content
            else:
                content_text = str(final_content)
        else:
            content_text = "No content available"
    
    # Add to history if not already there (avoid duplicates)
    if not st.session_state.content_history or st.session_state.content_history[-1]['content'] != content_text:
        version_number = len(st.session_state.content_history) + 1
        st.session_state.content_history.append({
            'version': version_number,
            'type': version_type,
            'content': content_text
        })
    
    return content_text

# Helper function to display content history/comparison
def display_content_comparison():
    """Display content history with comparison view"""
    if len(st.session_state.content_history) == 1:
        # Only original content
        st.subheader("📄 Generated Content")
        with st.expander(f"Version {st.session_state.content_history[0]['version']} ({st.session_state.content_history[0]['type']})", expanded=True):
            st.markdown(st.session_state.content_history[0]['content'])
    
    elif len(st.session_state.content_history) > 1:
        # Multiple versions - show comparison
        st.subheader("📊 Content Comparison")
        
        # Create tabs for each version
        tab_labels = [f"V{item['version']} ({item['type']})" for item in st.session_state.content_history]
        tabs = st.tabs(tab_labels)
        
        for i, (tab, content_item) in enumerate(zip(tabs, st.session_state.content_history)):
            with tab:
                st.markdown(content_item['content'])
        
        # Also show side-by-side comparison of latest two versions if more than one
        if len(st.session_state.content_history) >= 2:
            st.subheader("🔄 Side-by-Side Comparison")
            col1, col2 = st.columns(2)
            
            # Original vs Latest
            original = st.session_state.content_history[0]
            latest = st.session_state.content_history[-1]
            
            with col1:
                st.markdown(f"**{original['type']} (V{original['version']})**")
                with st.container(border=True):
                    st.markdown(original['content'])
            
            with col2:
                st.markdown(f"**{latest['type']} (V{latest['version']})**")
                with st.container(border=True):
                    st.markdown(latest['content'])

def main():
    st.set_page_config(
        page_title="Content Creator HIL",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Content Creator with Human-in-the-Loop")
    st.markdown("Create educational content with AI assistance and human approval")
    
    # Initialize session state
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = 'input'
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    if 'topic' not in st.session_state:
        st.session_state.topic = ""
    if 'content_history' not in st.session_state:
        st.session_state.content_history = []
    
    workflow = create_workflow()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key.startswith('workflow') or key in ['topic', 'content_history']:
                    del st.session_state[key]
            st.session_state.workflow_state = 'input'
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.content_history = []
            st.rerun()
        
        st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
        st.write(f"**Status:** {st.session_state.workflow_state}")
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Input phase
    if st.session_state.workflow_state == 'input':
        st.header("📝 Create Content")
        st.markdown("Enter a topic and our AI will create educational content suitable for 15-year-olds.")
        
        with st.form("content_form"):
            topic = st.text_input(
                "Enter a topic for educational content:",
                placeholder="e.g., Benefits of reading books, Solar energy basics, How photosynthesis works",
                value=st.session_state.get('topic', '')
            )
            submit = st.form_submit_button("🚀 Generate Content", type="primary")
        
        if submit and topic:
            st.session_state.topic = topic
            st.session_state.workflow_state = 'generating'
            st.rerun()
    
    # Generation phase
    elif st.session_state.workflow_state == 'generating':
        st.header("🔄 Generating Content...")
        
        with st.spinner("Our AI is creating educational content for you..."):
            try:
                # Reset content history for new topic
                st.session_state.content_history = []
                
                # Run the workflow - this will hit the interrupt in human_review
                result = workflow.invoke({"messages": st.session_state.topic}, config=config)
                st.session_state.workflow_result = result
                
                # Extract and store content in history
                extract_and_store_content(result, "Original")
                
                # Check if workflow was interrupted (human review needed)
                if '__interrupt__' in result:
                    st.session_state.workflow_state = 'human_review'
                    st.success("Content generated! Ready for human review.")
                else:
                    # Workflow completed without interruption (shouldn't happen with our graph)
                    st.session_state.workflow_state = 'completed'
                    st.success("Content completed!")
                
            except Exception as e:
                st.error(f"Error generating content: {e}")
                st.session_state.workflow_state = 'input'
                
        st.rerun()
    
    # Human review phase - handle the interrupt
    elif st.session_state.workflow_state == 'human_review':
        st.header("👤 Human Review")
        st.markdown("Please review the generated content and decide whether to approve it or request changes.")
        
        if st.session_state.workflow_result and '__interrupt__' in st.session_state.workflow_result:
            interrupt_data = st.session_state.workflow_result['__interrupt__'][0].value
            
            # Display the question
            st.subheader(interrupt_data['question'])
            
            # Display content comparison using our helper function
            display_content_comparison()
            
            st.subheader("Your Decision:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Approve Content", type="primary", use_container_width=True):
                    st.session_state.human_decision = "approved"
                    st.session_state.workflow_state = 'processing_decision'
                    st.rerun()
            
            with col2:
                if st.button("🔄 Request Changes", use_container_width=True):
                    st.session_state.workflow_state = 'request_changes'
                    st.rerun()
    
    # Request changes phase
    elif st.session_state.workflow_state == 'request_changes':
        st.header("📝 Request Changes")
        st.markdown("Please specify what changes you'd like to the content.")
        
        # Show content comparison
        display_content_comparison()
        
        with st.form("changes_form"):
            feedback = st.text_area(
                "What changes would you like?",
                placeholder="e.g., Make it shorter, add more examples, use simpler language, focus more on practical applications, etc.",
                height=100
            )
            col1, col2 = st.columns(2)
            with col1:
                submit_changes = st.form_submit_button("📤 Submit Changes", type="primary")
            with col2:
                cancel = st.form_submit_button("❌ Cancel")
        
        if submit_changes and feedback:
            st.session_state.human_decision = feedback
            st.session_state.workflow_state = 'processing_decision'
            st.rerun()
        
        if cancel:
            st.session_state.workflow_state = 'human_review'
            st.rerun()
    
    # Processing decision phase
    elif st.session_state.workflow_state == 'processing_decision':
        decision = st.session_state.human_decision
        
        if decision == "approved":
            st.header("✅ Processing Approval...")
        else:
            st.header("🔄 Processing Changes...")
        
        with st.spinner("Processing your decision..."):
            try:
                # Resume the workflow with the human decision - exactly like notebook
                result = workflow.invoke(Command(resume=decision), config=config)
                st.session_state.workflow_result = result
                
                # If this was a refinement request, store the refined content
                if decision != "approved":
                    extract_and_store_content(result, "Refined")
                
                # Check if workflow was interrupted again (after refinement)
                if '__interrupt__' in result:
                    st.session_state.workflow_state = 'human_review'
                    st.success("Content refined! Ready for another review.")
                else:
                    # Workflow completed
                    st.session_state.workflow_state = 'completed'
                    st.success("Process completed!")
                
            except Exception as e:
                st.error(f"Error processing decision: {e}")
                st.session_state.workflow_state = 'human_review'
                
        st.rerun()
    
    # Completed phase
    elif st.session_state.workflow_state == 'completed':
        st.header("✅ Content Approved!")
        st.success("Your content has been successfully created and approved.")
        
        if st.session_state.workflow_result:
            # Extract final content from the workflow result
            final_messages = st.session_state.workflow_result.get('messages', [])
            if final_messages:
                final_content = final_messages[-1]
                
                st.subheader("📄 Final Content:")
                with st.expander("View Final Content", expanded=True):
                    if hasattr(final_content, 'content'):
                        st.markdown(final_content.content)
                        content_text = final_content.content
                    else:
                        st.markdown(str(final_content))
                        content_text = str(final_content)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Copy Content", use_container_width=True):
                        st.code(content_text)
                        st.info("Content displayed above for copying")
                
                with col2:
                    st.download_button(
                        label="📥 Download",
                        data=content_text,
                        file_name=f"content_{st.session_state.topic.replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col3:
                    if st.button("🆕 Create New Content", use_container_width=True):
                        # Reset relevant session state
                        st.session_state.workflow_state = 'input'
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.workflow_result = None
                        st.session_state.topic = ""
                        st.session_state.content_history = []
                        if 'human_decision' in st.session_state:
                            del st.session_state.human_decision
                        st.rerun()

if __name__ == "__main__":
    main()