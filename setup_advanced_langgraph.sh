#!/bin/bash
# setup_advanced_langgraph.sh

echo "🚀 Setting up Advanced LangGraph Hierarchical Multi-Agent System"

# Create virtual environment
python -m venv langgraph_advanced_env
source langgraph_advanced_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core LangGraph dependencies..."
pip install langgraph>=0.6.0
pip install langchain-core>=0.3.0
pip install langchain-openai>=0.2.0
pip install langchain-anthropic>=0.2.0

# Install database dependencies
echo "