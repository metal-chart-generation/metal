prompt_template = """You are an expert Python developer who specializes in writing matplotlib code based on a given picture. 
I found a very nice picture in a STEM paper, but there is no corresponding source code available. 
I need your help to generate the Python code that can reproduce the picture based on the picture I provide.
\nNote that it is necessary to use figsize={figsize} to set the image size to match the original size.
\nNow, please give me the matplotlib code that reproduces the picture below.
"""