code_revision_prompt = """You are a professional data scientist tasked with revising the input code.

Objective:
The input code contains TODO comments that need to be addressed.
Please carefully review the code and make the necessary revisions to address the TODO comments.
Each comment might need more than one line of code to address.
Match the other lines of code style and structure in the input code.
Ensure that the revised code is correct and functional.
Return the FULL revised code to ensure the code is ready for the next stage of development. 

Response Format:
Full Code of the Revised Version: Present the revised code with the changes made to address the TODO comments.

Code to Revise:
```python
{code}
```
"""