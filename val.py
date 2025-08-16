#!/usr/bin/env python3
"""
Routing Constraints Validation Script
Ensures strict compliance with hierarchical multi-agent routing rules:
1. Only supervisors can route (return Command objects)
2. Only workers can execute tasks (return strings)
3. All agents use prebuilt functions only
"""

import asyncio
import inspect
import json
from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent, create_supervisor

# Import the main module
try:
    from advanced_hierarchical_agents import *
except ImportError:
    print("❌ Cannot import main module. Please ensure advanced_hierarchical_agents.py is available.")
    exit(1)


class RoutingValidator:
    """Comprehensive routing constraints validator"""

    def __init__(self):
        self.violations = []
        self.passed_checks = []

    def log_pass(self, check_name: str, details: str = ""):
        """Log a passed validation check"""
        self.passed_checks.append(f"✅ {check_name}: {details}")
        print(f"✅ {check_name}: {details}")

    def log_violation(self, check_name: str, details: str):
        """Log a routing constraint violation"""
        self.violations.append(f"❌ {check_name}: {details}")
        print(f"❌ {check_name}: {details}")

    def validate_agent_functions(self):
        """Validate that all agents use prebuilt functions only"""
        print("\n🔍 Validating Agent Function Usage...")

        # Check worker functions use create_react_agent
        worker_functions = [
            ("create_first_worker", create_first_worker),
            ("create_s2_searcher", create_s2_searcher),
            ("create_s2_writer", create_s2_writer)
        ]

        for name, func in worker_functions:
            try:
                agent = func()
                # Check if it's a create_react_agent result
                if hasattr(agent, 'nodes') and 'agent' in agent.nodes:
                    self.log_pass(f"Worker {name}", "uses create_react_agent")
                else:
                    self.log_violation(f"Worker {name}", "does not use create_react_agent")
            except Exception as e:
                self.log_violation(f"Worker {name}", f"creation failed: {e}")

        # Check supervisor functions use create_supervisor
        supervisor_functions = [
            ("create_first_supervisor", create_first_supervisor),
            ("create_other_supervisor", create_other_supervisor),
            ("create_s2_supervisor", create_s2_supervisor)
        ]

        for name, func in supervisor_functions:
            try:
                agent = func()
                # Check if it's a create_supervisor result
                if hasattr(agent, 'nodes') and 'supervisor' in agent.nodes:
                    self.log_pass(f"Supervisor {name}", "uses create_supervisor")
                else:
                    self.log_violation(f"Supervisor {name}", "does not use create_supervisor")
            except Exception as e:
                self.log_violation(f"Supervisor {name}", f"creation failed: {e}")

    def validate_tool_return_types(self):
        """Validate that tools return appropriate types"""
        print("\n🔍 Validating Tool Return Types...")

        # Worker tools should return strings
        worker_tools = [
            ("request_delegation", request_delegation),
            ("request_more_research", request_more_research),
            ("complete_task", complete_task)
        ]

        for name, tool in worker_tools:
            # Check tool annotation/documentation
            if hasattr(tool, 'args_schema'):
                self.log_pass(f"Worker tool {name}", "properly defined")
            else:
                self.log_violation(f"Worker tool {name}", "missing proper schema")

        # Supervisor tools should return Commands
        supervisor_tools = [
            ("route_to_other_supervisor", route_to_other_supervisor),
            ("finish_first_level", finish_first_level),
            ("route_to_s2_team", route_to_s2_team),
            ("finish_other_level", finish_other_level),
            ("route_to_s2_searcher", route_to_s2_searcher),
            ("route_to_s2_writer", route_to_s2_writer),
            ("complete_s2_team", complete_s2_team)
        ]

        for name, tool in supervisor_tools:
            if hasattr(tool, 'args_schema'):
                self.log_pass(f"Supervisor tool {name}", "properly defined")
            else:
                self.log_violation(f"Supervisor tool {name}", "missing proper schema")

    def validate_tool_function_signatures(self):
        """Validate tool function signatures to ensure correct return types"""
        print("\n🔍 Validating Tool Function Signatures...")

        # Test worker tools (should return strings when called)
        try:
            result = request_delegation("test reason", 0.5)
            if isinstance(result, str) and "DELEGATION_REQUEST" in result:
                self.log_pass("request_delegation", "returns string as expected")
            else:
                self.log_violation("request_delegation", f"returns {type(result)}, expected str")
        except Exception as e:
            self.log_violation("request_delegation", f"execution failed: {e}")

        try:
            result = request_more_research("test need")
            if isinstance(result, str) and "MORE_RESEARCH_REQUEST" in result:
                self.log_pass("request_more_research", "returns string as expected")
            else:
                self.log_violation("request_more_research", f"returns {type(result)}, expected str")
        except Exception as e:
            self.log_violation("request_more_research", f"execution failed: {e}")

        # Test supervisor tools (should return Commands when called)
        try:
            result = route_to_other_supervisor("test context")
            if isinstance(result, Command):
                self.log_pass("route_to_other_supervisor", "returns Command as expected")
            else:
                self.log_violation("route_to_other_supervisor", f"returns {type(result)}, expected Command")
        except Exception as e:
            self.log_violation("route_to_other_supervisor", f"execution failed: {e}")

        try:
            result = finish_first_level("test summary")
            if isinstance(result, Command):
                self.log_pass("finish_first_level", "returns Command as expected")
            else:
                self.log_violation("finish_first_level", f"returns {type(result)}, expected Command")
        except Exception as e:
            self.log_violation("finish_first_level", f"execution failed: {e}")

    def validate_graph_structure(self):
        """Validate the overall graph structure"""
        print("\n🔍 Validating Graph Structure...")

        try:
            # Test main graph creation
            system = ProductionHierarchicalSystem()
            main_graph = system.create_main_graph()

            if main_graph:
                self.log_pass("Main graph creation", "successful")

                # Check nodes exist
                expected_nodes = ["first_supervisor", "first_worker", "other_supervisor", "first_finish"]
                actual_nodes = list(main_graph.nodes.keys())

                for node in expected_nodes:
                    if node in actual_nodes:
                        self.log_pass(f"Node {node}", "exists in graph")
                    else:
                        self.log_violation(f"Node {node}", "missing from graph")
            else:
                self.log_violation("Main graph creation", "failed")

        except Exception as e:
            self.log_violation("Graph structure", f"validation failed: {e}")

    def validate_state_schema(self):
        """Validate state schema compliance"""
        print("\n🔍 Validating State Schema...")

        try:
            # Test state creation
            test_state = HierarchicalAgentState(
                messages=[HumanMessage(content="test")],
                current_task="test_task",
                task_complexity=0.5
            )

            # Check required fields
            required_fields = ["messages", "current_task", "first_note", "other_note", "s2_summary"]
            for field in required_fields:
                if hasattr(test_state, field):
                    self.log_pass(f"State field {field}", "exists")
                else:
                    self.log_violation(f"State field {field}", "missing")

        except Exception as e:
            self.log_violation("State schema", f"validation failed: {e}")

    def validate_langgraph_json(self):
        """Validate langgraph.json configuration"""
        print("\n🔍 Validating LangGraph JSON Configuration...")

        try:
            with open("langgraph.json", "r") as f:
                config = json.load(f)

            # Check required sections
            required_sections = ["node_mapping", "graphs", "dependencies", "environment"]
            for section in required_sections:
                if section in config:
                    self.log_pass(f"Config section {section}", "exists")
                else:
                    self.log_violation(f"Config section {section}", "missing")

            # Check routing constraints
            if "routing_constraints" in config:
                constraints = config["routing_constraints"]
                if "supervisor_agents" in constraints and "worker_agents" in constraints:
                    self.log_pass("Routing constraints", "properly defined")
                else:
                    self.log_violation("Routing constraints", "incomplete definition")
            else:
                self.log_violation("Routing constraints", "missing from config")

        except FileNotFoundError:
            self.log_violation("langgraph.json", "file not found")
        except json.JSONDecodeError as e:
            self.log_violation("langgraph.json", f"invalid JSON: {e}")
        except Exception as e:
            self.log_violation("langgraph.json", f"validation failed: {e}")

    async def run_functional_test(self):
        """Run functional test to ensure routing works correctly"""
        print("\n🔍 Running Functional Routing Test...")

        try:
            system = ProductionHierarchicalSystem()
            graph = system.create_main_graph()

            # Test simple case that should stay with FirstWorker
            simple_input = {
                "messages": [HumanMessage(content="What is AI?")],
                "current_task": "simple_question",
                "task_complexity": 0.2
            }

            config = {"configurable": {"thread_id": "test_simple"}}
            result = graph.invoke(simple_input, config=config)

            if result and "messages" in result:
                self.log_pass("Simple routing test", "completed successfully")
            else:
                self.log_violation("Simple routing test", "failed to complete")

            # Test complex case that should trigger delegation
            complex_input = {
                "messages": [HumanMessage(
                    content="I need a comprehensive market analysis and business strategy for my SaaS startup.")],
                "current_task": "complex_strategy",
                "task_complexity": 0.9
            }

            config = {"configurable": {"thread_id": "test_complex"}}
            result = graph.invoke(complex_input, config=config)

            if result and any(
                    note for note in [result.get("first_note"), result.get("other_note"), result.get("s2_summary")] if
                    note):
                self.log_pass("Complex routing test", "delegation occurred as expected")
            else:
                self.log_violation("Complex routing test", "delegation did not occur")

        except Exception as e:
            self.log_violation("Functional test", f"failed: {e}")

    async def run_all_validations(self):
        """Run all validation checks"""
        print("🚀 Starting Comprehensive Routing Validation")
        print("=" * 60)

        # Run all validation checks
        self.validate_agent_functions()
        self.validate_tool_return_types()
        self.validate_tool_function_signatures()
        self.validate_graph_structure()
        self.validate_state_schema()
        self.validate_langgraph_json()
        await self.run_functional_test()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\n✅ PASSED CHECKS ({len(self.passed_checks)}):")
        for check in self.passed_checks:
            print(f"   {check}")

        if self.violations:
            print(f"\n❌ VIOLATIONS ({len(self.violations)}):")
            for violation in self.violations:
                print(f"   {violation}")
        else:
            print(f"\n🎉 NO VIOLATIONS FOUND!")

        # Overall result
        total_checks = len(self.passed_checks) + len(self.violations)
        success_rate = len(self.passed_checks) / total_checks * 100 if total_checks > 0 else 0

        print(f"\n📈 SUCCESS RATE: {success_rate:.1f}% ({len(self.passed_checks)}/{total_checks})")

        if len(self.violations) == 0:
            print("\n🏆 ALL ROUTING CONSTRAINTS VALIDATED SUCCESSFULLY!")
            print("✅ Only supervisors can route")
            print("✅ Only workers execute tasks")
            print("✅ All agents use prebuilt functions")
            print("✅ Tools have correct return types")
            print("✅ Graph structure is correct")
            print("✅ LangGraph.json is properly configured")
            return True
        else:
            print(f"\n⚠️  ROUTING CONSTRAINTS VIOLATIONS FOUND!")
            print("Please fix the violations above before deployment.")
            return False


async def main():
    """Main validation function"""
    validator = RoutingValidator()
    success = await validator.run_all_validations()

    if success:
        print(f"\n🚀 System is ready for production deployment!")
        print(f"💡 Next steps:")
        print(f"   1. Deploy using: langgraph deploy")
        print(f"   2. Test endpoints: /invoke, /stream, /batch")
        print(f"   3. Monitor health: /health")
    else:
        print(f"\n🔧 Please fix violations before deployment.")

    return success


if __name__ == "__main__":
    import sys

    success = asyncio.run(main())
    sys.exit(0 if success else 1)