prompts_template = """You are a professional data scientist tasked to critique the generated chart against a reference chart to improve the code for generating the chart.

Objective: 
There is (are) issue(s) in the generated chart that diverge from the reference chart regarding the {lowest_metric}. Compare the original chart (left) and the generated chart (right) in the provided image.
Observe the differences between the reference chart and the generated chart, and provide a detailed critique of the generated chart.
Insert TODO comments directly into the input code to indicate necessary changes. Place the TODO comment above the line of code that requires modification. Be specific and practical in the TODO comments. Avoid generic suggestions or additions unrelated to the critique.

Instructions:
- Step1-2: 
Avoid commenting on unrelated metrics or general stylistic choices unless they directly affect the {lowest_metric}. 
Only critique on elements that in reference chart but not in generated chart. 
For example, if the reference chart doesn't have a title, don't critique the title in the generated chart.
Here are insturction for the chart critique:
{chart_instructions}
- Step3-4:
If the chart critique specifies particular color values, include a TODO comment in the code to remove the opacity setting (e.g. alpha). This ensures the color is accurately replaced with the specified values from the critique
If the chart critique identifies incorrect or missing text (of label, annotation, etc.), please include the correct text in the TODO comment for easy reference.
If the chart critique identifies the mismatched type of chart, please add TODO comments above the type-related functions calls to correct the chart type. 


Response Format:
1. Chart Critique: Issues in the generated chart that diverge from the reference chart. Be specific and detailed.
2. Code Critique: Provide a critique of the code that generated the chart. Identify the issues and suggest improvements by adding TODO comments.
3. Full Code with added TODO Comments: Present the input code with the TODO comments added above the relevant lines. Do not modify the code directly.

Code to Comment On:
```python
{code}
```
"""

metric_instructions = {
    "text": """Please first identify the text elements and location in the chart, including the title, axis labels, axis values, axis ticks( interval, exact values ), and legends.
Then, compare the text elements in the generated chart with the reference chart and provide a detailed critique.""",
    
    "color": """Here are the dominant colors in the reference chart: {reference_colors}.
    Here are the dominant colors in the generated chart: {generated_colors}.
    Find the mismatched colors in the generated chart compared to the reference chart and provide a detailed critique.""",
    
     "overall": """Please provide a detailed critique of how the generated chart diverges from the reference chart concerning the overall appearance and style.
    Compare the overall appearance and style of the generated chart with the reference chart and provide a detailed critique.""",
}  