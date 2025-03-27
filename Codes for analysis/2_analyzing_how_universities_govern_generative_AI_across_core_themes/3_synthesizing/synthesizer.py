from openai import OpenAI
import os




def evaluate(common_policies,sections,api_key = "<insert openAI API key>", model="gpt-4o-mini", max_tokens=5000):
    client = OpenAI(api_key=api_key)
    system_message = """
    You are an evaluator grading if the provided analysis is grounded in the original document
    """
    concatenated_sections = "\n\n---\n\n".join(sections)
    task_prompt=f'''
    You previously generated a concise set of common AI policies from a set of distinct outputs. Your task if to ensure that no new policies, facts, or explanations were massively hallucinated. 
    Strictly return ONLY a "PASS" the analysis is grounded and does not contain any newly hallucinated policies, facts, or explanations. 
    Return "FAIL" with a detailed evaluation if there are any massive hallucinations. 
    Here is the original set of AI policies, with each section marked by a by "---": {concatenated_sections}

    Here is the analysis: {common_policies}

    Note that minor potentially missing items or intepretative nuances does not constitute a hallucination. Hallucinations occur when completely new facts were made up. 

    '''
    # API call setup
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task_prompt}
    ]

    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0
    )

    # Return the response
    return response.choices[0].message.content

def reflection(common_policies,sections,evaluation, api_key = "<insert openAI API key>", model="gpt-4o-mini", max_tokens=5000):
    client = OpenAI(api_key=api_key)
    system_message = """
    You are an assistant reflecting and revising on a given evaluation of a generated analysis. 
    """
    concatenated_sections = "\n\n---\n\n".join(sections)
    task_prompt=f'''
    You previously evaluated an analysis which sought determine a common set of overlapping AI policies. 
    You have identified the following concerns surrounding potential hallucinations: {evaluation}

    Here is the analysis: {common_policies}

    Here is the original set of AI policies, with each section marked by a by "---": {concatenated_sections}

    Reflect and revise the analysis based on the evaluation. Strictly return the revised analysis only. 


    '''
    # API call setup
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task_prompt}
    ]

    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0
    )
    return response.choices[0].message.content

def get_common_policies(sections, query, api_key, model, max_tokens):
    """
    Identifies, merges, and ranks common policies from multiple model outputs, excluding outliers.
    """
    client = OpenAI(api_key=api_key)

    # Combine sections into one input
    concatenated_sections = "\n\n---\n\n".join(sections)

    # Define the system message
    system_message = """
    You are assisting a university administrator in consolidating AI policies based on analyses across multiple models.
    Focus on identifying the most prevalent policies and keep the output concise.
    """

    task_prompt = f"""
    The following text contains **5 synthesized analyses** of AI policies across universities.
    These analyses, from 5 separate models, aim to highlight common policies addressing AI usage in relation to the given query: {query} 
    Each section, marked by "---", represents the output from a different model.

    {concatenated_sections}

    **Your task:**

    1. **Identify and Merge Common Policies:**  
       - Focus only on policies that appear in majority of the outputs.
       - If policies have similar meaning but different wording, **consolidate them into a single, clearly worded policy.**
       - Ensure each final policy is distinct and non-overlapping.

    2. **Rank Policies by Consistency:**  
       - Rank policies based on their frequency across outputs, with top-ranked policies being the most consistently mentioned.
       - For ties, rank by overall consistency.

    3. **Consolidate and Exclude:**  
       - Include only policies that meet the majority criterion (present in at least 3 outputs).
       - Use the following format for each policy:
         - **Policy**: <Title of policy>  
         - **Description**: <Consolidated description>  
         - **Examples**: <Example institutions implementing this policy>

    **Important Requirements:**  
    - **Use only the information provided**. Do not introduce new information, speculate, or make assumptions.
    - **Avoid hallucinating or fabricating details**. Only include information directly supported by the text.
    - Present the final policies as a ranked, numbered list, starting with the most frequently mentioned policy.
    - Ensure a professional, clear, and accurate presentation.

    **Output Format Example:**  
    - Provide a numbered list, starting with the most frequently mentioned policy.

    """

    # API call setup
    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": task_prompt.strip()}
    ]

    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0
    )

    # Return the response
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


def synthesize_common_policies(sections, query, api_key = "<insert openAI API key>", model="gpt-4o-mini", max_tokens=1500):
    common_policies=get_common_policies(sections, query, api_key, model, max_tokens)
    evaluation=evaluate(common_policies,sections)
    if evaluation=="PASS":
        print("Hallucination check: PASS")
        reflected_policies=common_policies
    else:
        print(evaluation)
        print("Hallucination failed!")
        reflected_policies=reflection(common_policies,sections,evaluation)

    print("-"*100)
    return reflected_policies