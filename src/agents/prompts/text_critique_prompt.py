critique_to_guidelines_prompt = """You are a professional data scientist tasked with analyzing input code and adding targeted TODO comments based on the provided critique.

Instructions:
Please identify whether there is(are) issue(s) in the input code that need to be addressed based on the visual critique, and give concrete feedback.
Offer clear and actionable suggestions to address the identified issues.
Insert TODO comments directly into the input code to indicate necessary changes.
Place the TODO comment above the line of code that requires modification.
Be specific and practical in the TODO comments. Avoid generic suggestions or additions unrelated to the critique.
Do not make changes beyond adding TODO comments to the code, such as change existing code or add new lines of code.
Do not modify the code or add comments about code style, unrelated improvements, or hypothetical enhancements outside the critique's scope.
Do not mention reference charts in the TODO comments, making the comment self-contained.

Special Instructions:
If the critique specifies particular color values, include a TODO comment in the code to remove the opacity setting (e.g. alpha). This ensures the color is accurately replaced with the specified values from the critique
If the critique identifies incorrect or missing text (of label, annotation, etc.), please include the correct text in the TODO comment for easy reference.
If the critique identifies the mismatched type of chart, please add TODO comments above the type-related functions calls to correct the chart type. 

Response Format:
1. Issues: Summarize the issues identified in the critique.
2. Suggestions: Provide specific suggestions for addressing the issues.
3. Full Code with added TODO Comments: Present the input code with the TODO comments added above the relevant lines. Please ONLY add TODO comments to the input code, do not modify the code in any other way.

Critique:
{critique}

Code to Comment On:
```python
{code}
```

Note: If you believe there is no issue, please respond with SKIP.
"""
