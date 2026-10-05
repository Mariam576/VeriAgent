"""Long-term memory: what VeriAgent keeps between conversations.

Three kinds of memory exist in this project, and they are deliberately
different mechanisms:

    short-term   the turns of one conversation (``messages`` table, app/api/chat.py)
    profile      stable preferences, stored as key/value rows (``user_memory``)
    semantic     lasting facts in free text, found by meaning (``semantic_memory``)

A fourth thing is often confused with memory: the checkpointer. It saves the
state of ONE investigation so it can pause and resume; it is deleted when the
investigation ends.
"""
