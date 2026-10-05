"""Evaluation suite: a fixed set of test investigations with known right answers.

    python -m app.evaluation.run_eval            run every case and print the metrics
    python -m app.evaluation.run_eval --help     options (one category, one case, real database)

Concept - evaluation:
    Unit tests check that the code does what the programmer wrote. An
    evaluation checks whether the agent's ANSWERS are right. Each case has
    documents, a question and the expected outcome (which page must be cited,
    which contradiction must be found, what the calculator must return...).
    Running all cases gives numbers such as "contradiction detection 100%" or
    "hallucinated citations: 0", and those numbers tell you whether a change to
    a prompt, a model or the retrieval settings made the agent better or worse.
"""
