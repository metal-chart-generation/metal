prompts_template = """You are a professional data scientist tasked with evaluating a generated chart against a reference chart.

Objective: 
Please identify whether there is (are) issue(s) in the generated chart that diverge from the reference chart concerning the {lowest_metric}, and give concrete feedback.
Compare the original chart (left) and the generated chart (right) in the provided image.
Provide a detailed critique of how the generated chart diverges from the reference chart concerning the {lowest_metric}.
Avoid commenting on unrelated metrics or general stylistic choices unless they directly affect the {lowest_metric}.
Only critique on elements that in reference chart but not in generated chart. 
For example, if the reference chart doesn't have a title, don't critique the title in the generated chart.

Instructions:
{instructions}

Response Format:
1. Observation (Reference Chart): Identify the chart elements in the reference chart.
2. Observation (Generated Chart): Identify the chart elements in the generated chart.
3. Critique: Issues in the generated chart that diverge from the reference chart concerning the {lowest_metric}. Be specific and detailed. If there is numeric value or text, please provide the exact value or text.

Note: If you believe there is no issue, please respond with SKIP.
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
