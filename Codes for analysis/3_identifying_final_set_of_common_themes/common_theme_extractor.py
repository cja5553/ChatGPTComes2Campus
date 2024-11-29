from openai import OpenAI
import os

def parse_sections(input_text):
    """
    Split the input into sections delimited by "---".
    This will handle edge cases such as multiple or missing delimiters.
    """
    sections = [section.strip() for section in input_text.split('---') if section.strip()]
    return sections

def get_common_themes(sections, api_key, model="gpt-4", max_tokens=1500):
    """
    Send the sections to the model to identify common themes,
    removing outliers and ranking them based on consistency in model outputs.
    """
    client = OpenAI(
        api_key=api_key,
    )
    
    # Concatenate all sections into one input for the model
    concatenated_sections = "\n\n---\n\n".join(sections)
    
    # System message for the task
    system_message = """
    You are a university administrator reviewing synthesized AI policy themes from multiple models.
    Your goal is to consolidate and rank the common themes across these outputs.
    Please keep the final output concise and focused on the most prevalent themes.
    """
    
    task_prompt = f"""
    The following text contains outputs from **5 different models** that synthesized AI policies across universities.
    Each section, delimited by "---", represents an output from a different model:

    {concatenated_sections}

    **Your task:**

    1. **Identify Common and Similar Themes:**
    - Examine the themes across all 5 outputs.
    - **Include themes that appear in the majority of the outputs (i.e., in at least 3 out of 5 models).**
    - If a theme appears in different forms but has similar meaning or focus (e.g., risks, limitations, and ethical concerns), **merge them into a single theme** with a consolidated explanation.
    - Ensure each final theme is distinct and does not overlap with others.

    2. **Rank the Themes:**
    - Rank the themes based on how consistently they appear at the top across different model outputs.
    - Themes that frequently appear at the top should be ranked higher.
    - If themes have the same frequency at the top positions, rank them based on overall consistency across the outputs.

    3. **Minimize and Consolidate:**
    - Exclude themes that do not meet the majority criterion.
    - Condense and synthesize the information for each theme.

    **For each theme, provide:**

    1. **Key Point**: A short, succinct statement (5-8 words) summarizing the theme.
    - *Example*: "Privacy and security in generative AI use".

    2. **Explanation**: A detailed explanation of the key point, highlighting the commonalities in how universities address this issue.

    3. **Examples**: Specific examples showing how different universities implement this policy.
    - *Example*: "Universities A and B require students to document AI tool usage, while Universities X and Y prohibit AI assistance in assignments".

    **Important Notes:**

    - **Merge themes** that have the same or similar core ideas, even if they are worded differently.
    - **Base your response solely on the provided synthesized information.**
    - **Do not introduce any new information or make assumptions.**
    - **Present the themes in a ranked, numbered list, starting with the most frequently mentioned theme.**
    - **Ensure professional and clear presentation.**

    Please present the final ranked list of key themes below.
    """

    
    # Messages list for API call
    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": task_prompt.strip()}
    ]
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens, 
        temperature=0.1  # Low temperature for consistency and minimalism
    )
    
    # Extract and return the result
    return response.choices[0].message.content






def clean_output(output):
    """
    Optionally clean the output for better formatting and consistency.
    This can be used if needed to further process the model's response.
    """
    # Remove extra line breaks or trailing/leading spaces
    cleaned_output = output.strip()
    cleaned_output = cleaned_output.replace('\n\n', '\n').replace('\n---\n', '\n')
    return cleaned_output


def read_file(file_name='../results/interim_results/key_points.txt'):
    # Open the file in read mode
    with open(file_name, 'r') as file:
        data = file.read()  # Read the entire file content
    
    # Print the file content
    return(data)
    


