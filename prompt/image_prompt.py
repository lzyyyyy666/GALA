# Previous prompt kept for reference.
# SubGraphPrompt = """A Bug Report is provided together with a focused visual graph summary extracted from the bug scenario images.
# The visual graph may describe frontend UI components, diagram structures, or natural-image scene nodes and edges.
#
# Use the issue statement, focused visual graph summary, and reranked files to generate *SEARCH/REPLACE* edits to fix the issue. Use the suggested solution from the Bug Report directly to fix the bug.
# Prefer the smallest effective change set that fully resolves the issue, and avoid unnecessary edits outside the minimal fix scope.
#
# Treat the reranked files listed below as the primary starting point for inspection and modification. Prioritize those files first, and only expand beyond them if the Bug Report makes it necessary to complete the fix correctly.
# After identifying a candidate fix, carefully verify it against the Bug Report to ensure the modification is complete, correct, and fully addresses the described behavior rather than only partially matching the reranked files.
#
# INPUT:
#
# * Bug Report
# '''
# {{problem_statement}}
# '''
#
# * Focused Visual Graph Summary
# '''
# {{focused_visual_graph_summary}}
# '''
#
# * Reranked Files
# '''
# {{localization_guidance}}
# '''
#
# If reranked files are available, use them to guide your initial inspection and edits. You can start with those files before searching more broadly.
# Use the suggested solution from the Bug Report directly to fix the bug!
# Conduct a careful analysis to ensure your patch is executable and effectively resolves the stated problem!"""

SubGraphPrompt = """You are fixing a real repository bug. Use the Bug Report as the primary specification, inspect the codebase, and make the smallest correct code change that resolves the reported behavior.

Your goal is to produce a real repository fix, not a speculative explanation. The final result should be an executable code change that can be captured as a git diff.

Decision priority:
1. Bug Report and expected behavior
2. Actual code logic and call flow in the repository
3. Reranked files as a starting point for inspection
4. Focused visual graph summary as auxiliary context only

Rules:
- First identify the likely root cause in code before editing.
- Make the smallest code change that fully fixes the reported behavior; avoid unrelated refactors, cleanup edits, or formatting-only changes.
- Use the Bug Report to infer intended behavior, but verify the fix against actual code logic rather than assuming the report already specifies the exact solution.
- Treat reranked files as starting hints only; if they are insufficient, continue searching only as far as needed to complete the fix correctly.
- For visual or UI bugs, use the focused visual graph summary only to understand the affected elements or relationships; do not let it override code evidence.

INPUT:

* Bug Report
'''
{{problem_statement}}
'''

* Focused Visual Graph Summary
'''
{{focused_visual_graph_summary}}
'''

* Reranked Files
'''
{{localization_guidance}}
'''

* Suggested Edit Targets
'''
{{edit_targets_guidance}}
'''

* Previous Failure Reason
'''
{{previous_failure_reason}}
'''

Working approach:
- Start inspection from the reranked files if they are relevant.
- Check the suggested edit targets early and decide whether those locations actually need modification.
- Use the previous failure reason to avoid repeating known failing patterns.
- Trace the code path responsible for the reported behavior.
- Edit only the files necessary for a correct fix.
- Before finishing, check that the patch is coherent, minimal, and directly tied to the Bug Report.

Produce the code changes needed to fix the issue."""


ImagePrompt = """A Bug Report is provided together with a focused visual graph summary extracted from the bug scenario images.
The visual graph may describe frontend UI components, diagram structures, or natural-image scene nodes and edges.

Please localize the bug based on the issue statement and output bug-related code snippets.
INPUT:

* Bug Report
'''
{{problem_statement}}
'''

* Focused Visual Graph Summary
'''
{{focused_visual_graph_summary}}
'''

Conduct a careful analysis to ensure your localization result is correct!"""


VLMPrompt = """You are a master at analyzing images and code.

# Task
I will provide you with a bug report, an image related to the bug (image resolution={{resolution}}), and possibly the code snippet(s) that correspond to the bug. Your job is to analyze both the bug description and the code snippet(s), locate the region in the image that is most relevant to this bug, and return its bounding-box coordinates.

# Input

* Bug Report
'''
{{problem_statement}}
'''

* Code snippets
'''
{{code_snips}}
'''

# Output format
<reason>
Please describe your reasoning process.
</reason>
<result>Return the coordinates of the relevant region in the form [x, y, w, h], where x and y are the coordinates of the top-left corner of the bounding box, and w and h are its width and height.</result>"""
