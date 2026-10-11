"""Extractive summarization with LLM prompting.

One engine for every dataset: an *experiment* config names a task (what the
model selects), a prompt set (how it is asked) and the data; a *grid* names the
technique variants; a *model* alias plus a machine profile names where it runs.
"""

__version__ = "1.0.0"
