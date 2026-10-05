"""Advanced document comparison (Version 3).

Version 1 and 2 compare two quantities across documents: the total amount and
the payment terms. This package compares the documents themselves:

    profile.py   the structured summary of one document (its "profile")
    rules.py     build a profile with rules (fallback and offline mode)
    llm.py       build a profile with the LLM, then verify every field in code
    dates.py     read dates in the formats documents use
    names.py     decide whether two company names mean the same company
    engine.py    the deterministic checks that compare profiles
    store.py     where profiles are kept, and how they are built on demand
"""
