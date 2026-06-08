"""Validate agent configuration files for consistency.
Usage: python validate_agent_config.py <config.json>
"""
import json
import sys


def validate_config(config: dict) -> list[str]:
    errors = []
    agents = config.get("agents", [])

    if not agents:
        errors.append("No agents defined")
        return errors

    agent_names = set()
    for agent in agents:
        name = agent.get("name")
        if not name:
            errors.append("Agent missing 'name' field")
            continue

        if name in agent_names:
            errors.append(f"Duplicate agent name: {name}")
        agent_names.add(name)

        # Check required fields
        for field in ["role", "description"]:
            if not agent.get(field):
                errors.append(f"Agent '{name}' missing '{field}'")

        # Check tools exist (if specified)
        tools = agent.get("tools", [])
        if not isinstance(tools, list):
            errors.append(f"Agent '{name}' tools should be a list")

        # Check handoff targets exist
        handoffs = agent.get("handoff_targets", [])
        for target in handoffs:
            if target not in [a.get("name") for a in agents]:
                errors.append(f"Agent '{name}' handoff target '{target}' not defined")

    # Check for orphan agents (no one hands off to them and they're not the entry point)
    entry = config.get("entry_agent")
    if entry and entry not in agent_names:
        errors.append(f"Entry agent '{entry}' not found in agents list")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_agent_config.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    errors = validate_config(config)

    if errors:
        print(f"Agent config validation FAILED for {sys.argv[1]}:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        agents = config.get("agents", [])
        print(f"Agent config validation PASSED ({len(agents)} agents)")
        for agent in agents:
            print(f"  - {agent['name']}: {agent.get('role', 'N/A')}")


if __name__ == "__main__":
    main()
