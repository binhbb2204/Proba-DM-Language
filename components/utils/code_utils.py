import re
import tkinter as tk

def apply_syntax_highlighting(editor, content, pattern, tag, multiline=True):
    """Apply highlighting for a pattern"""
    flags = 0 if multiline else re.MULTILINE
    for match in re.finditer(pattern, content, flags):
        start = f"1.0 + {match.start()} chars"
        end = f"1.0 + {match.end()} chars"
        editor.tag_add(tag, start, end)