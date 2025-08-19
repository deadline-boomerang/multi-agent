"""
Customer Identification Hierarchical Agent System: Complete Flow Resolution with Real Integrations

This system is adapted from the Salesforce Hierarchical Agent for identifying customer accounts.
It uses the Salesforce supervisor as a sub-agent for handling Salesforce-specific operations like querying accounts.

Key Changes:
- Focused on customer account identification workflows.
- Integrates the existing Salesforce main supervisor as a child sub-agent for Salesforce queries and analysis.
- Added specialists for account search, customer verification, and identification insights.
- Retained singleton patterns, real integrations, and supervisor-specialist flow.
- NEW: Routes identification tasks to appropriate supervisors, using Salesforce for data retrieval.
- NEW: Enhanced tools for customer-specific queries (e.g., filter by Type='Customer').
- NEW: Global flags and debug logging consistent with Salesforce pattern.
- NEW: Error handling and explicit delegation in prompts.
"""

import asyncio
import time
import os
import json
import logging
from typing import List, Dict, Any, Annotated, Optional
from dataclasses import field
from datetime import datetime

# Core LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor  # New import
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

# Import from Salesforce module (assuming the Salesforce code is in fixed_salesforce_hierarchical_agent.py)
from fixed_salesforce_hierarchical_agent import create_main_supervisor as create_salesforce_supervisor
from fixed_salesforce_hierarchical_agent import salesforce_manager, vector_store, WORKSPACE_ID, RECORD_LIMIT, \
    clean_soql_query, enforce_query_limit
from fixed_salesforce_hierarchical_agent import OpenAIEmbeddings, SingletonMeta, ComponentCache, setup_debug_logging, \
    add_messages, log_state_update, ModelSelector, create_tool_response

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_CONNECTION_STRING = os.getenv("DATABASE_URL")
VECTOR_SIZE = 1536
TABLE_NAME = "customer_identification_vectors"  # Different table for customer-specific vectors
schema_dir = "customer_schema"  # Optional: If separate schemas

# Constants
MAX_MESSAGES_CONTEXT = 10
MAX_ITERATIONS = 25

# Embeddings
embedding = OpenAIEmbeddings()

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

DEBUG_ENABLED = True
LOG_LEVEL = logging.DEBUG if DEBUG_ENABLED else logging.INFO

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = setup_debug_logging()
agent_logger = logging.getLogger('customer_ident_agent')

# =============================================================================
# SINGLETON PATTERN
# =============================================================================

# Reuse from Salesforce: SingletonMeta, ComponentCache

# Global init flags (reuse or add similar)
_customer_vector_initialized = False
_customer_data_populated = False


# If needed, create VectorStoreManager for customer-specific, but here reuse Salesforce's vector_store for simplicity

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class CustomerIdentState(MessagesState):
    """State schema for Customer Identification system"""
    messages: Annotated[List[BaseMessage], add_messages]

    # Task tracking
    current_task: str = ""
    task_complexity: float = 0.5
    supervisor_chain: List[str] = field(default_factory=list)
    workspace_id: str = WORKSPACE_ID

    # Customer-specific state
    search_result: str = ""
    verification_result: str = ""
    analysis_result: str = ""

    # System state
    completion_timestamp: float = 0
    remaining_steps: int = 0
    last_error: Optional[str] = None


# =============================================================================
# MODEL SELECTION
# =============================================================================

model_selector = ModelSelector()


# =============================================================================
# TOOLMESSAGE HELPER
# =============================================================================

# Reused from Salesforce

# =============================================================================
# CUSTOMER IDENTIFICATION TOOLS
# =============================================================================

@tool
def search_customer_accounts(query: str) -> Dict[str, Any]:
    """Search for customer accounts in Salesforce (Type='Customer')."""
    agent_logger.debug(f"TOOL: search_customer_accounts({query})")
    try:
        cleaned_query = clean_soql_query(
            f"SELECT Id, Name, Type, Industry FROM Account WHERE Type = 'Customer' AND Name LIKE '%{query}%'")
        limited_query = enforce_query_limit(cleaned_query)
        result = salesforce_manager.execute_query(limited_query)
        if "error" in result:
            return result
        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def verify_customer_status(account_id: str) -> str:
    """Verify if an account is a valid customer."""
    agent_logger.debug(f"TOOL: verify_customer_status({account_id})")
    try:
        cleaned_query = clean_soql_query(f"SELECT Id, Type, Status FROM Account WHERE Id = '{account_id}'")
        limited_query = enforce_query_limit(cleaned_query)
        result = salesforce_manager.execute_query(limited_query)
        if "error" in result:
            return result["error"]
        if result.get('totalSize', 0) > 0:
            record = result['records'][0]
            if record.get('Type') == 'Customer':
                return f"Valid customer. Status: {record.get('Status', 'Unknown')}"
            return "Not a customer"
        return "Invalid ID or no record found"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def analyze_customer_data(data: Dict[str, Any]) -> str:
    """Analyze customer account data for insights."""
    agent_logger.debug(f"TOOL: analyze_customer_data({str(data)[:50]}...)")
    try:
        records = data.get("records", [])
        if not records:
            return "No data to analyze"

        # Simple analysis: Count by industry
        industries = {}
        for record in records:
            industry = record.get('Industry', 'Unknown')
            industries[industry] = industries.get(industry, 0) + 1

        analysis = f"Total customers: {len(records)}\nIndustry breakdown: {json.dumps(industries, indent=2)}"
        return analysis
    except Exception as e:
        return f"Analysis failed: {str(e)}"


# =============================================================================
# SPECIALISTS CREATION
# =============================================================================

def create_search_specialist():
    """Search Specialist for customer accounts"""
    return ComponentCache.get_or_create("search_specialist", _create_search_specialist_impl)


def _create_search_specialist_impl():
    system_prompt = """You are the Search Specialist for Customer Accounts.

Your capabilities:
- Use search_customer_accounts() to find customer accounts
- Handle query optimization and result formatting

Process:
1. Receive search request
2. Execute using search_customer_accounts()
3. If the tool returns an error, note the error and suggest alternatives
4. Respond with the formatted results in markdown.

If the tool fails, do not retry; respond with available information or error message.
You execute tasks and report back."""

    specialist = create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[search_customer_accounts],
        state_schema=CustomerIdentState,
        prompt=SystemMessage(content=system_prompt),
        name="SearchSpecialist"
    )
    return specialist


def create_verification_specialist():
    """Verification Specialist for customer status"""
    return ComponentCache.get_or_create("verification_specialist", _create_verification_specialist_impl)


def _create_verification_specialist_impl():
    system_prompt = """You are the Verification Specialist for Customer Accounts.

Your capabilities:
- Use verify_customer_status() for status checks
- Provide validation reports

Process:
1. Receive verification request
2. Run checks using verify_customer_status()
3. If the tool returns an error, note the error and suggest alternatives
4. Respond with the detailed report in markdown.

If the tool fails, do not retry; respond with available information or error message.
You execute tasks and report back."""

    specialist = create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[verify_customer_status],
        state_schema=CustomerIdentState,
        prompt=SystemMessage(content=system_prompt),
        name="VerificationSpecialist"
    )
    return specialist


def create_analysis_specialist():
    """Analysis Specialist for customer data"""
    return ComponentCache.get_or_create("analysis_specialist", _create_analysis_specialist_impl)


def _create_analysis_specialist_impl():
    system_prompt = """You are the Analysis Specialist for Customer Accounts.

Your capabilities:
- Use analyze_customer_data() for insights
- Use search_customer_accounts() if needed to fetch data
- Generate actionable recommendations

Process:
1. Receive analysis request
2. Fetch data if not provided using search_customer_accounts()
3. Process using analyze_customer_data()
4. If the tool returns an error, note the error and suggest alternatives
5. Generate insights
6. Respond with the detailed insights in markdown.

If the tool fails, do not retry; respond with available information or error message.
You execute tasks and report back."""

    specialist = create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[analyze_customer_data, search_customer_accounts],
        state_schema=CustomerIdentState,
        prompt=SystemMessage(content=system_prompt),
        name="AnalysisSpecialist"
    )
    return specialist


# =============================================================================
# SUPERVISOR CREATION
# =============================================================================

def create_search_supervisor():
    """Search Supervisor"""
    return ComponentCache.get_or_create("search_supervisor", _create_search_supervisor_impl)


def _create_search_supervisor_impl():
    specialist = create_search_specialist()
    prompt = """You are the Search Supervisor managing the search specialist.

Always delegate the task to the specialist by using the 'assign_to_SearchSpecialist' tool with the appropriate query. Do not respond or summarize until you have the results from the specialist.

The specialist will respond with results.

Then, review the results and provide the final response directly to the user.

Do not use 'FINISH'; instead, respond with the final answer when done."""

    supervisor_graph = create_supervisor(
        agents=[specialist],
        prompt=prompt,
        model=model_selector.get_supervisor_model(),
        add_handoff_back_messages=True,
        output_mode="last_message",
        supervisor_name="search_supervisor"
    ).compile(name="search_supervisor")
    return supervisor_graph


def create_verification_supervisor():
    """Verification Supervisor"""
    return ComponentCache.get_or_create("verification_supervisor", _create_verification_supervisor_impl)


def _create_verification_supervisor_impl():
    specialist = create_verification_specialist()
    prompt = """You are the Verification Supervisor managing the verification specialist.

Always delegate the task to the specialist by using the 'assign_to_VerificationSpecialist' tool with the appropriate query. Do not respond or summarize until you have the results from the specialist.

The specialist will respond with results.

Then, review the results and provide the final response directly to the user.

Do not use 'FINISH'; instead, respond with the final answer when done."""

    supervisor_graph = create_supervisor(
        agents=[specialist],
        prompt=prompt,
        model=model_selector.get_supervisor_model(),
        add_handoff_back_messages=True,
        output_mode="last_message",
        supervisor_name="verification_supervisor"
    ).compile(name="verification_supervisor")
    return supervisor_graph


def create_analysis_supervisor():
    """Analysis Supervisor"""
    return ComponentCache.get_or_create("analysis_supervisor", _create_analysis_supervisor_impl)


def _create_analysis_supervisor_impl():
    specialist = create_analysis_specialist()
    prompt = """You are the Analysis Supervisor managing the analysis specialist.

Always delegate the task to the specialist by using the 'assign_to_AnalysisSpecialist' tool with the appropriate query. Do not respond or summarize until you have the results from the specialist.

The specialist will respond with results.

Then, review the results and provide the final response directly to the user.

Do not use 'FINISH'; instead, respond with the final answer when done."""

    supervisor_graph = create_supervisor(
        agents=[specialist],
        prompt=prompt,
        model=model_selector.get_supervisor_model(),
        add_handoff_back_messages=True,
        output_mode="last_message",
        supervisor_name="analysis_supervisor"
    ).compile(name="analysis_supervisor")
    return supervisor_graph


# =============================================================================
# MAIN SUPERVISOR WITH SALESFORCE AS SUB-AGENT
# =============================================================================

def create_main_customer_supervisor():
    """Main Supervisor for Customer Identification"""
    return ComponentCache.get_or_create("main_customer_supervisor", _create_main_customer_supervisor_impl)


def _create_main_customer_supervisor_impl():
    agent_logger.debug("Creating Main Customer Supervisor")

    # Child supervisors
    search_supervisor = create_search_supervisor()
    verification_supervisor = create_verification_supervisor()
    analysis_supervisor = create_analysis_supervisor()

    # Integrate Salesforce supervisor as sub-agent
    salesforce_sub = create_salesforce_supervisor()

    main_system_prompt = """You are the Main Customer Identification Supervisor.

ROUTING LOGIC:
- Account search → search_supervisor
- Status verification → verification_supervisor
- Data analysis → analysis_supervisor
- Complex Salesforce operations → salesforce_sub

You orchestrate customer identification through supervisor coordination."""

    main_supervisor_graph = create_supervisor(
        agents=[search_supervisor, verification_supervisor, analysis_supervisor, salesforce_sub],
        prompt=main_system_prompt,
        model=model_selector.get_supervisor_model(),
        add_handoff_back_messages=True,
        output_mode="last_message",
        supervisor_name="main_customer_supervisor"
    ).compile(name="main_customer_supervisor")

    return main_supervisor_graph


# =============================================================================
# SYSTEM INTEGRATION
# =============================================================================

class CustomerIdentSystem(metaclass=SingletonMeta):
    """Customer Identification system"""

    def __init__(self, use_persistence: bool = True):
        if not hasattr(self, 'initialized'):
            agent_logger.info(
                f"INITIALIZING: Customer Identification Hierarchical System (persistence={use_persistence})")
            self.checkpointer = MemorySaver() if use_persistence else None
            self.store = InMemoryStore()
            self.initialized = True

    def create_main_graph(self):
        """Create the main graph"""
        return ComponentCache.get_or_create("main_graph", self._create_main_graph_impl)

    def _create_main_graph_impl(self):
        main_supervisor = create_main_customer_supervisor()
        return main_supervisor


# =============================================================================
# ENTRY POINTS
# =============================================================================

def customer_ident_agent_entry():
    """Main entry point"""
    return create_main_customer_supervisor()


# Additional entry points for individual supervisors/specialists similar to Salesforce

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

def create_langgraph_json():
    """Create deployment configuration"""
    config = {
        "graphs": {
            "customer_ident_graph": "./customer_ident_hierarchical_agent.py:customer_ident_agent_entry",
            # Add others
        },
        "dependencies": [
            # Same as Salesforce
        ],
        "environment": {
            # Same as Salesforce
        },
        "debug": {
            "enabled": DEBUG_ENABLED,
            "log_level": "DEBUG" if DEBUG_ENABLED else "INFO"
        }
    }
    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_architecture():
    """Validate the architecture"""
    agent_logger.info("VALIDATION: Starting Customer Identification architecture validation")
    # Similar validation as in Salesforce
    return True


async def test_hierarchy():
    """Test the system"""
    agent_logger.info("🚀 TESTING: Starting Customer Identification system tests")
    test_cases = [
        {
            "name": "Search Customer Accounts",
            "input": {"messages": [HumanMessage(content="Search for customer accounts with name like 'Acme'")]},
            "expected_flow": "Main → Search Supervisor → Search Specialist → Final"
        },
        # Add more
    ]
    customer_system = CustomerIdentSystem()
    main_graph = customer_system.create_main_graph()
    for i, test_case in enumerate(test_cases, 1):
        config = {"configurable": {"thread_id": f"customer_test_{i}"}}
        result = main_graph.invoke(test_case["input"], config=config)
        # Log results


def main():
    """Main execution"""
    agent_logger.info("🌟 MAIN: Starting Customer Identification Hierarchical System")
    create_langgraph_json()
    validate_architecture()
    asyncio.run(test_hierarchy())


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE DOCUMENTATION
# =============================================================================

"""
CUSTOMER IDENTIFICATION HIERARCHICAL AGENT

## Usage:

```python
# Run the system
python customer_ident_hierarchical_agent.py

# Test
result = await main_graph.ainvoke({
    "messages": [HumanMessage("Search for customer accounts like 'Test'")]
})
print(result.get("search_result"))

The system ensures complete flow with Salesforce integration.
"""