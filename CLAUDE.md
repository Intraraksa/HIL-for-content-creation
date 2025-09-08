# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph-based project that implements human-in-the-loop (HIL) content creation workflows using Google's Gemini AI. The project demonstrates interactive AI agents that can create and refine content with human approval steps.

## Development Environment

- **Python Version**: 3.13+ (specified in `.python-version`)
- **Package Manager**: Uses `uv` for dependency management (indicated by `uv.lock`)
- **Environment Variables**: Configure API keys in `.env` file

## Core Dependencies

- **LangGraph**: State-based AI workflow orchestration
- **LangChain**: LLM integration and prompt management
- **Google GenAI**: Primary LLM provider (Gemini models)
- **FastAPI**: Web framework for potential API endpoints
- **Streamlit**: For interactive web interfaces
- **Chroma**: Vector database for embeddings

## Project Architecture

The main implementation is in `notebook.ipynb` which demonstrates:

1. **State Management**: Uses TypedDict-based state with message history
2. **Agent System**: 
   - `content_creator`: Creates content suitable for kids under 15
   - `refine_creator`: Refines content with professional tone
   - `human_approve`: Implements human-in-the-loop approval with interrupts
3. **Workflow Graph**: StateGraph connecting agents with conditional routing
4. **Checkpointing**: MemorySaver for conversation persistence

## Development Commands

```bash
# Install dependencies
uv sync

# Run main Python script
python main.py

# Start Jupyter notebook for development
jupyter notebook

# Run Streamlit app (if implemented)
streamlit run <app_file>

# Run FastAPI server (if implemented)
uvicorn <app>:app --reload
```

## Key Patterns

- State is passed between agents using the `State` TypedDict
- Human approval uses LangGraph's `interrupt()` function for workflow pauses
- Commands control workflow routing with `goto` directives
- Checkpointing enables conversation resumption with thread IDs
- Human in the loop for content creation. If human approved it then go to END node. on the other hand, if human need to rewrite content. It will go to refine_creator node.