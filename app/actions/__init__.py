"""Actions outside VeriAgent, carried out through MCP after a person approved them (Version 3).

Concept - MCP (Model Context Protocol):
    A standard way for an application to use tools that live in another
    program. The other program is an "MCP server": it lists its tools (name,
    description, the shape of the arguments) and carries out calls. The
    application is the "MCP client". Because the protocol is the same
    everywhere, VeriAgent talks to GitHub's MCP server and to its own little
    ticket server with exactly the same code; only the address differs.

        VeriAgent  ->  MCP client  ->  MCP server  ->  GitHub (or a ticket file)

Concept - why approval comes first:
    Everything VeriAgent did before this was reading and calculating. Creating
    an issue changes something in another system and cannot be taken back by
    the agent. So the agent only PROPOSES it, the investigation pauses, and a
    person sees the exact text before anything is sent (spec section 29:
    interrupt before external communication and irreversible tools).

    client.py         connect to an MCP server, find the tool, call it
    ticket_server.py  a small MCP server of our own, so this works without any account
    proposals.py      what the agent proposes: the issue text, built from the report
    runner.py         carry out an approved action exactly once, and keep a record
"""
