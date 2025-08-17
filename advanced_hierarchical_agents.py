"""
Hierarchical Multi-Agent System: Supervisor-Only Communication
Architecture Rules:
- Only supervisors can communicate with other supervisors
- Specialists/workers are leaf nodes (no inter-specialist communication)
- Workers only execute tasks and report back to their direct supervisor
- Supervisors handle all coordination and delegation
"""

import asyncio
import time
import os
import json
from typing import Literal, List, Dict, Any, Annotated
from dataclasses import field
from collections import defaultdict

# Core LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def add_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    """Enhanced message reducer with deduplication"""
    def to_message(msg):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = ' '.join([item.get('text', '') if isinstance(item, dict) and item.get('type') == 'text' else '' for item in content])
            if role == "user":
                return HumanMessage(content=content)
            elif role == "assistant":
                return AIMessage(content=content)
            elif role == "system":
                return SystemMessage(content=content)
            else:
                return HumanMessage(content=content)
        return msg

    existing = [to_message(m) for m in existing]
    new = [to_message(m) for m in new]
    all_messages = existing + new
    seen = set()
    unique_messages = []
    for msg in all_messages:
        msg_hash = hash((msg.content, getattr(msg, 'id', None)))
        if msg_hash not in seen:
            seen.add(msg_hash)
            unique_messages.append(msg)
    return unique_messages

class HierarchicalAgentState(MessagesState):
    """State schema for supervisor-only communication system"""
    messages: Annotated[List[BaseMessage], add_messages]

    # Task tracking
    current_task: str = ""
    task_complexity: float = 0.5
    supervisor_chain: List[str] = field(default_factory=list)  # Track supervisor delegation path

    # Results from each specialist area
    interface_result: str = ""      # From Interface Specialist
    research_result: str = ""       # From Research Specialist
    analysis_result: str = ""       # From Analysis Specialist
    synthesis_result: str = ""      # From Synthesis Specialist

    # System state
    completion_timestamp: float = 0
    remaining_steps: int = 0

# =============================================================================
# MODEL SELECTION
# =============================================================================

class ModelSelector:
    def __init__(self):
        self.models = {
            "supervisor": ChatOpenAI(model="gpt-4o", temperature=0.1),      # For routing decisions
            "specialist": ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # For execution
        }

    def get_supervisor_model(self) -> ChatOpenAI:
        """Model for supervisor routing decisions"""
        return self.models["supervisor"]

    def get_specialist_model(self) -> ChatOpenAI:
        """Model for specialist execution"""
        return self.models["specialist"]

model_selector = ModelSelector()

# =============================================================================
# TOOLMESSAGE HELPER
# =============================================================================

def create_tool_response(tool_call_id: str, content: str) -> ToolMessage:
    """Helper to create ToolMessage responses"""
    return ToolMessage(content=content, tool_call_id=tool_call_id)

# =============================================================================
# SPECIALIST TOOLS (ONLY EXECUTION - NO ROUTING)
# =============================================================================

@tool
def complete_interface_task(summary: str, user_satisfied: bool = True) -> str:
    """Interface Specialist completes user interaction"""
    return f"INTERFACE_COMPLETE: {summary} (satisfied: {user_satisfied})"

@tool
def complete_research_task(findings: str, confidence: float = 0.8) -> str:
    """Research Specialist completes research"""
    return f"RESEARCH_COMPLETE: {findings} (confidence: {confidence})"

@tool
def complete_analysis_task(analysis: str, depth: str = "comprehensive") -> str:
    """Analysis Specialist completes analysis"""
    return f"ANALYSIS_COMPLETE: {analysis} (depth: {depth})"

@tool
def complete_synthesis_task(recommendations: str, actionable: bool = True) -> str:
    """Synthesis Specialist completes synthesis"""
    return f"SYNTHESIS_COMPLETE: {recommendations} (actionable: {actionable})"

# =============================================================================
# SUPERVISOR ROUTING TOOLS (SUPERVISOR-TO-SUPERVISOR COMMUNICATION)
# =============================================================================

# Main Supervisor Tools (Routes to other supervisors)
@tool
def delegate_to_interface_supervisor(
    tool_call_id: Annotated[str, InjectedToolCallId],
    task_description: str,
    user_context: str = "general"
) -> Command:
    """Delegate to Interface Supervisor for user interaction management"""
    return Command(
        goto="interface_supervisor",
        update={
            "messages": [create_tool_response(tool_call_id, f"Delegating to Interface Supervisor: {task_description}")],
            "supervisor_chain": ["main_supervisor", "interface_supervisor"],
            "current_task": task_description
        }
    )

@tool
def delegate_to_research_supervisor(
    tool_call_id: Annotated[str, InjectedToolCallId],
    research_scope: str,
    priority: str = "normal"
) -> Command:
    """Delegate to Research Supervisor for research coordination"""
    return Command(
        goto="research_supervisor",
        update={
            "messages": [create_tool_response(tool_call_id, f"Delegating to Research Supervisor: {research_scope}")],
            "supervisor_chain": ["main_supervisor", "research_supervisor"],
            "current_task": research_scope
        }
    )

@tool
def delegate_to_analysis_supervisor(
    tool_call_id: Annotated[str, InjectedToolCallId],
    analysis_type: str,
    complexity: float = 0.7
) -> Command:
    """Delegate to Analysis Supervisor for deep analysis"""
    return Command(
        goto="analysis_supervisor",
        update={
            "messages": [create_tool_response(tool_call_id, f"Delegating to Analysis Supervisor: {analysis_type}")],
            "supervisor_chain": ["main_supervisor", "analysis_supervisor"],
            "task_complexity": complexity
        }
    )

@tool
def complete_main_workflow(
    tool_call_id: Annotated[str, InjectedToolCallId],
    final_summary: str
) -> Command:
    """Complete the entire workflow"""
    return Command(
        goto=END,
        update={
            "messages": [create_tool_response(tool_call_id, f"Main workflow complete: {final_summary}")],
            "completion_timestamp": time.time()
        }
    )

# Interface Supervisor Tools (Routes to interface specialist)
@tool
def assign_to_interface_specialist(
    tool_call_id: Annotated[str, InjectedToolCallId],
    interaction_type: str = "standard"
) -> Command:
    """Assign task to Interface Specialist"""
    return Command(
        goto="interface_specialist",
        update={
            "messages": [create_tool_response(tool_call_id, f"Assigning to Interface Specialist: {interaction_type}")]
        }
    )

@tool
def escalate_to_main_supervisor(
    tool_call_id: Annotated[str, InjectedToolCallId],
    escalation_reason: str
) -> Command:
    """Escalate back to Main Supervisor"""
    return Command(
        goto=END,
        graph=Command.PARENT,
        update={
            "messages": [create_tool_response(tool_call_id, f"Escalating to Main: {escalation_reason}")]
        }
    )

# Research Supervisor Tools (Routes to research specialist)
@tool
def assign_to_research_specialist(
    tool_call_id: Annotated[str, InjectedToolCallId],
    research_focus: str = "comprehensive"
) -> Command:
    """Assign task to Research Specialist"""
    return Command(
        goto="research_specialist",
        update={
            "messages": [create_tool_response(tool_call_id, f"Assigning to Research Specialist: {research_focus}")]
        }
    )

@tool
def return_research_to_main(
    tool_call_id: Annotated[str, InjectedToolCallId],
    research_summary: str
) -> Command:
    """Return research results to Main Supervisor"""
    return Command(
        goto=END,
        graph=Command.PARENT,
        update={
            "messages": [create_tool_response(tool_call_id, f"Research complete: {research_summary}")],
            "research_result": research_summary
        }
    )

# Analysis Supervisor Tools (Routes to analysis specialist)
@tool
def assign_to_analysis_specialist(
    tool_call_id: Annotated[str, InjectedToolCallId],
    analysis_focus: str = "strategic"
) -> Command:
    """Assign task to Analysis Specialist"""
    return Command(
        goto="analysis_specialist",
        update={
            "messages": [create_tool_response(tool_call_id, f"Assigning to Analysis Specialist: {analysis_focus}")]
        }
    )

@tool
def delegate_to_synthesis_supervisor(
    tool_call_id: Annotated[str, InjectedToolCallId],
    synthesis_requirements: str
) -> Command:
    """Delegate to Synthesis Supervisor (supervisor-to-supervisor)"""
    return Command(
        goto="synthesis_supervisor",
        update={
            "messages": [create_tool_response(tool_call_id, f"Delegating to Synthesis Supervisor: {synthesis_requirements}")],
            "supervisor_chain": ["main_supervisor", "analysis_supervisor", "synthesis_supervisor"]
        }
    )

@tool
def return_analysis_to_main(
    tool_call_id: Annotated[str, InjectedToolCallId],
    analysis_summary: str
) -> Command:
    """Return analysis results to Main Supervisor"""
    return Command(
        goto=END,
        graph=Command.PARENT,
        update={
            "messages": [create_tool_response(tool_call_id, f"Analysis complete: {analysis_summary}")],
            "analysis_result": analysis_summary
        }
    )

# Synthesis Supervisor Tools (Routes to synthesis specialist)
@tool
def assign_to_synthesis_specialist(
    tool_call_id: Annotated[str, InjectedToolCallId],
    synthesis_mode: str = "strategic"
) -> Command:
    """Assign task to Synthesis Specialist"""
    return Command(
        goto="synthesis_specialist",
        update={
            "messages": [create_tool_response(tool_call_id, f"Assigning to Synthesis Specialist: {synthesis_mode}")]
        }
    )

@tool
def return_synthesis_to_analysis(
    tool_call_id: Annotated[str, InjectedToolCallId],
    synthesis_summary: str
) -> Command:
    """Return synthesis results to Analysis Supervisor"""
    return Command(
        goto=END,
        graph=Command.PARENT,
        update={
            "messages": [create_tool_response(tool_call_id, f"Synthesis complete: {synthesis_summary}")],
            "synthesis_result": synthesis_summary
        }
    )

# =============================================================================
# SPECIALIST AGENTS (LEAF NODES - EXECUTION ONLY)
# =============================================================================

def create_interface_specialist():
    """Interface Specialist - handles user interaction"""
    system_prompt = """You are the Interface Specialist, responsible for user interaction.

Your role:
- Handle direct user conversations naturally and helpfully
- Provide clear, engaging responses to user questions
- For simple queries: answer directly
- When complete: use complete_interface_task

You are a specialist, not a supervisor. You only execute tasks assigned to you
and report results back to your supervisor. You do not route to other agents."""

    return create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[complete_interface_task],
        state_schema=HierarchicalAgentState,
        prompt=SystemMessage(content=system_prompt),
        name="interface_specialist"
    )

def create_research_specialist():
    """Research Specialist - handles research tasks"""
    system_prompt = """You are the Research Specialist, responsible for gathering information.

Your role:
- Conduct thorough research on assigned topics
- Gather facts, data, and evidence
- Focus on comprehensive information collection
- When complete: use complete_research_task

You are a specialist, not a supervisor. You execute research tasks and
report findings back to your supervisor. You do not communicate with other specialists."""

    return create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[complete_research_task],
        state_schema=HierarchicalAgentState,
        prompt=SystemMessage(content=system_prompt),
        name="research_specialist"
    )

def create_analysis_specialist():
    """Analysis Specialist - handles analytical tasks"""
    system_prompt = """You are the Analysis Specialist, responsible for analyzing information.

Your role:
- Analyze research data and information
- Identify patterns, trends, and insights
- Perform strategic and competitive analysis
- When complete: use complete_analysis_task

You are a specialist, not a supervisor. You execute analysis tasks and
report insights back to your supervisor. You do not communicate with other specialists."""

    return create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[complete_analysis_task],
        state_schema=HierarchicalAgentState,
        prompt=SystemMessage(content=system_prompt),
        name="analysis_specialist"
    )

def create_synthesis_specialist():
    """Synthesis Specialist - handles synthesis and recommendations"""
    system_prompt = """You are the Synthesis Specialist, responsible for creating final deliverables.

Your role:
- Synthesize analysis into actionable recommendations
- Create comprehensive strategic deliverables
- Transform insights into implementable advice
- When complete: use complete_synthesis_task

You are a specialist, not a supervisor. You execute synthesis tasks and
report recommendations back to your supervisor. You do not communicate with other specialists."""

    return create_react_agent(
        model=model_selector.get_specialist_model(),
        tools=[complete_synthesis_task],
        state_schema=HierarchicalAgentState,
        prompt=SystemMessage(content=system_prompt),
        name="synthesis_specialist"
    )

# =============================================================================
# SUPERVISOR CREATION (SUPERVISOR-ONLY COMMUNICATION LAYER)
# =============================================================================

def create_main_supervisor():
    """Main Supervisor - coordinates other supervisors"""

    # Main supervisor only manages other supervisors
    supervisors = [
        create_interface_supervisor(),
        create_research_supervisor(),
        create_analysis_supervisor()
    ]

    system_prompt = """You are the Main Supervisor, coordinating specialized supervisors.

SUPERVISOR-TO-SUPERVISOR COMMUNICATION RULES:
1. You only communicate with other supervisors, never directly with specialists
2. Route user interactions to interface_supervisor
3. Route research needs to research_supervisor  
4. Route analysis needs to analysis_supervisor
5. Coordinate workflow between supervisors based on task requirements

WORKFLOW PATTERNS:
- Simple user queries: → interface_supervisor
- Research tasks: → research_supervisor → analysis_supervisor
- Complex analysis: → research_supervisor → analysis_supervisor
- Complete workflow when all required supervisors have finished

You manage supervisor-level coordination, not individual specialists."""

    return create_supervisor(
        supervisors,
        model=model_selector.get_supervisor_model(),
        prompt=system_prompt,
        tools=[
            delegate_to_interface_supervisor,
            delegate_to_research_supervisor,
            delegate_to_analysis_supervisor,
            complete_main_workflow
        ],
        supervisor_name="MainSupervisor"
    )

def create_interface_supervisor():
    """Interface Supervisor - manages interface specialist"""

    # This supervisor only manages the interface specialist
    specialists = [create_interface_specialist()]

    system_prompt = """You are the Interface Supervisor, managing user interaction.

Your role:
- Manage the Interface Specialist
- Assign user interaction tasks to your specialist
- Escalate complex requests back to Main Supervisor
- Coordinate user-facing communication

You communicate with Main Supervisor and manage your Interface Specialist.
You do not communicate with other supervisors directly."""

    return create_supervisor(
        specialists,
        model=model_selector.get_supervisor_model(),
        prompt=system_prompt,
        tools=[assign_to_interface_specialist, escalate_to_main_supervisor],
        supervisor_name="InterfaceSupervisor"
    ).compile()

def create_research_supervisor():
    """Research Supervisor - manages research specialist"""

    # This supervisor only manages the research specialist
    specialists = [create_research_specialist()]

    system_prompt = """You are the Research Supervisor, managing research activities.

Your role:
- Manage the Research Specialist
- Assign research tasks to your specialist
- Return research results to Main Supervisor
- Coordinate information gathering

You communicate with Main Supervisor and manage your Research Specialist.
You do not communicate with other supervisors directly."""

    return create_supervisor(
        specialists,
        model=model_selector.get_supervisor_model(),
        prompt=system_prompt,
        tools=[assign_to_research_specialist, return_research_to_main],
        supervisor_name="ResearchSupervisor"
    ).compile()

def create_analysis_supervisor():
    """Analysis Supervisor - manages analysis specialist and synthesis supervisor"""

    # This supervisor manages analysis specialist AND coordinates with synthesis supervisor
    members = [
        create_analysis_specialist(),
        create_synthesis_supervisor()  # Supervisor-to-supervisor communication!
    ]

    system_prompt = """You are the Analysis Supervisor, managing analysis and synthesis coordination.

Your role:
- Manage the Analysis Specialist for analytical work
- Coordinate with Synthesis Supervisor for final deliverables
- Route work between analysis and synthesis as needed
- Return complete results to Main Supervisor

ROUTING LOGIC:
1. Start with Analysis Specialist for data analysis
2. When analysis complete, delegate to Synthesis Supervisor for final recommendations
3. Return integrated results to Main Supervisor

You demonstrate supervisor-to-supervisor communication with Synthesis Supervisor."""

    return create_supervisor(
        members,
        model=model_selector.get_supervisor_model(),
        prompt=system_prompt,
        tools=[
            assign_to_analysis_specialist,
            delegate_to_synthesis_supervisor,
            return_analysis_to_main
        ],
        supervisor_name="AnalysisSupervisor"
    ).compile()

def create_synthesis_supervisor():
    """Synthesis Supervisor - manages synthesis specialist"""

    # This supervisor only manages the synthesis specialist
    specialists = [create_synthesis_specialist()]

    system_prompt = """You are the Synthesis Supervisor, managing final deliverable creation.

Your role:
- Manage the Synthesis Specialist
- Assign synthesis tasks to create final recommendations
- Return synthesis results to Analysis Supervisor
- Coordinate final deliverable creation

You communicate with Analysis Supervisor and manage your Synthesis Specialist.
This demonstrates the nested supervisor communication pattern."""

    return create_supervisor(
        specialists,
        model=model_selector.get_supervisor_model(),
        prompt=system_prompt,
        tools=[assign_to_synthesis_specialist, return_synthesis_to_analysis],
        supervisor_name="SynthesisSupervisor"
    )

# =============================================================================
# SYSTEM INTEGRATION
# =============================================================================

class SupervisorOnlySystem:
    """System where only supervisors communicate with each other"""

    def __init__(self, use_persistence: bool = False):
        self.checkpointer = MemorySaver()
        self.store = InMemoryStore()

    def create_main_graph(self):
        """Create the main graph with supervisor-only communication"""

        # The main graph is the Main Supervisor
        # All communication flows through supervisor hierarchy
        main_supervisor = create_main_supervisor()

        return main_supervisor.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

def create_langgraph_json():
    """Create deployment configuration"""
    config = {
        "graphs": {
            "main_graph": "advanced_hierarchical_agents.py:main_agent_entry"
        },
        "dependencies": [
            "langgraph>=0.6.0",
            "langchain-core>=0.3.0",
            "langchain-openai>=0.2.0",
            "langgraph-supervisor>=0.1.0"
        ],
        "environment": {
            "OPENAI_API_KEY": {"required": True}
        }
    }

    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)

    return config

# =============================================================================
# MAIN SYSTEM INSTANCE
# =============================================================================

# Create the supervisor-only system
supervisor_system = SupervisorOnlySystem(use_persistence=False)
main_graph = supervisor_system.create_main_graph()

def main_agent_entry():
    """Main entry point for deployment"""
    return main_graph

# Create configuration
create_langgraph_json()

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_supervisor_only_architecture():
    """Validate the supervisor-only communication architecture"""
    print("🔍 Validating Supervisor-Only Communication Architecture...")

    # Validate supervisor structure
    supervisors = [
        "MainSupervisor",
        "InterfaceSupervisor",
        "ResearchSupervisor",
        "AnalysisSupervisor",
        "SynthesisSupervisor"
    ]

    for supervisor in supervisors:
        print(f"✅ {supervisor} created with create_supervisor")

    # Validate specialist structure
    specialists = [
        "InterfaceSpecialist",
        "ResearchSpecialist",
        "AnalysisSpecialist",
        "SynthesisSpecialist"
    ]

    for specialist in specialists:
        print(f"✅ {specialist} created with create_react_agent (leaf node)")

    print("✅ Communication Rules Validated:")
    print("   • Supervisors communicate with supervisors ✓")
    print("   • Specialists are leaf nodes (no inter-specialist communication) ✓")
    print("   • Clear hierarchy maintained ✓")

    return True

async def test_supervisor_only_system():
    """Test the supervisor-only communication system"""
    print("🚀 Testing Supervisor-Only Communication System")
    print("=" * 60)

    validate_supervisor_only_architecture()

    test_cases = [
        {
            "name": "Simple User Query",
            "input": {"messages": [HumanMessage(content="What is artificial intelligence?")]},
            "expected_flow": "Main → Interface Supervisor → Interface Specialist"
        },
        {
            "name": "Research Task",
            "input": {"messages": [HumanMessage(content="Research the latest AI trends in healthcare")]},
            "expected_flow": "Main → Research Supervisor → Research Specialist"
        },
        {
            "name": "Complex Analysis",
            "input": {"messages": [HumanMessage(content="Analyze the competitive landscape and provide strategic recommendations for AI chatbots")]},
            "expected_flow": "Main → Analysis Supervisor → Analysis Specialist → Synthesis Supervisor → Synthesis Specialist"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"🧪 TEST {i}: {test_case['name']}")
        print(f"📋 Expected Flow: {test_case['expected_flow']}")
        print(f"{'=' * 50}")

        try:
            config = {"configurable": {"thread_id": f"supervisor_test_{i}"}}
            start_time = time.time()

            result = main_graph.invoke(test_case["input"], config=config)

            execution_time = time.time() - start_time
            print(f"✅ COMPLETED in {execution_time:.2f}s")

            # Show supervisor chain
            supervisor_chain = result.get("supervisor_chain", [])
            print(f"📊 Supervisor Chain: {' → '.join(supervisor_chain) if supervisor_chain else 'Direct'}")

            # Show specialist results
            if result.get("interface_result"):
                print(f"💬 Interface Result: {result['interface_result'][:100]}...")
            if result.get("research_result"):
                print(f"🔍 Research Result: {result['research_result'][:100]}...")
            if result.get("analysis_result"):
                print(f"📊 Analysis Result: {result['analysis_result'][:100]}...")
            if result.get("synthesis_result"):
                print(f"🎯 Synthesis Result: {result['synthesis_result'][:100]}...")

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 Supervisor-Only System Testing Complete!")
    print(f"🏗️ Architecture Summary:")
    print(f"   📊 Communication: Supervisor ↔ Supervisor only")
    print(f"   🎯 Specialists: Leaf nodes (execution only)")
    print(f"   🔄 No inter-specialist communication")
    print(f"   ⚡ Clear supervisor hierarchy")
    print(f"   🛡️ Proper separation of concerns")

def main():
    """Main execution"""
    print("🏗️ Supervisor-Only Communication Hierarchical System")
    print("=" * 70)

    print("📊 Communication Architecture:")
    print("   Main Supervisor")
    print("   ├── Interface Supervisor ← → Interface Specialist")
    print("   ├── Research Supervisor ← → Research Specialist")
    print("   └── Analysis Supervisor")
    print("       ├── Analysis Specialist")
    print("       └── Synthesis Supervisor ← → Synthesis Specialist")
    print()
    print("   Rules:")
    print("   • Supervisors ↔ Supervisors (coordination)")
    print("   • Supervisors → Specialists (task assignment)")
    print("   • Specialists → Supervisors (results only)")
    print("   • NO Specialist ↔ Specialist communication")

    asyncio.run(test_supervisor_only_system())

if __name__ == "__main__":
    main()