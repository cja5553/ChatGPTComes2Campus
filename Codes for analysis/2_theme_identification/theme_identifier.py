from openai import OpenAI
import json
import random


def load_dict(file_path):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Ensure the data is a dictionary
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        # If the data is a list of key-value pairs, convert it to a dictionary
        return {item[0]: item[1] for item in data}
    else:
        raise ValueError("Loaded data is not in a format that can be converted to a dictionary.")

def jumble_dict_order(dictionary):
    # Convert dictionary items to a list and shuffle the list
    items = list(dictionary.items())
    random.shuffle(items)
    
    # Re-create the dictionary with shuffled order
    jumbled_dict = dict(items)
    return jumbled_dict

def hallucinator_check(analysis, text):
    system_message = '''You are tasked with verifying the factual accuracy of an analysis based on provided documents.'''

    task_prompt = f'''You previously synthesized a list of themes and provided corresponding explainations and summarized examples from documents spanning AI policies in US universities. 
    Your task is to check your analysis, focusing on the references to specific institutes from **summarized examples**, for hallucinations. 
    We define hallucinations as scenarios where facts are made up or significantly inaccurate. 
    Discrepencies in interpretations, contextual nuances or omissions of certain details, particularly if irrelevant to the theme, should not be considered a hallucination. 

    If the content is not hallucinated, strictly respond with only the single word "PASS".
    If you find significant inaccuracies or fabrications, respond with "FAIL" and provide specific which aspect of your analysis was hallucinated, paying close attention to our aforementioned definition of hallucination.
    Here is the analysis: {analysis}

    The AI policy documents are provided in a dictionary format as such: {{"<University>:<summary of AI policies>"}}
    Here is the original documents: {text}. 

    Note: Minor paraphrasing or omissions that do not alter the overall meaning should not be considered a failure. 
    If the content is not hallucinated, strictly respond simply with only the single word "PASS".'''
    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]


def hallucinator_corrector(analysis, text, hallucination):
    # System message to set the context
    system_message = '''You are a professional assistant tasked with correcting an analysis of AI policy themes. Your goal is to address identified hallucinations or misrepresentations by referencing the original policy documents. Your responses should be accurate, neutral, and based solely on the provided information.'''

    # Task prompt with reference to previous step
    task_prompt = f'''You previously provided the following analysis of AI policy themes:

        {analysis}

        A reviewer has identified the following hallucinations or misrepresentations in your analysis:

        {hallucination}

        **Your task:**

        - **Correct the Analysis**: Revise the analysis to eliminate the identified inaccuracies, ensuring all information is accurate and directly supported by the original policy documents.
        - **Preserve the Format**: Maintain the same original analysis format for consistency.

        **Original Policy Summaries:**

        The AI policy summaries are provided below in a dictionary format, where each key is a university name and each value is the corresponding policy summary:

        {text}

        **Instructions:**

        - **Base Your Corrections Solely on the Provided Policy Summaries**: Do not introduce any new information or make assumptions.
        - **Reference Specific Universities**: When providing examples, mention the universities that correspond to the policies.
        - **Ensure Clarity and Accuracy**: Make sure the corrected analysis accurately reflects the content of the original policy summaries.

        **Analysis Format:**

        For each theme, present the information in the same format as provided:

        1. **Key Point**: A short, succinct statement (5-8 words) summarizing the theme. Example: "Privacy and security in generative AI use".

        2. **Explanation**: A detailed explanation of the key point, highlighting the commonalities in how universities address this issue.

        3. **Examples**: Specific examples showing how different universities implement this policy. For example, "Universities A and B require students to document AI tool usage, while Universities X and Y prohibit AI assistance in assignments".

        **Important Notes:**

        - **Do Not Include New Themes**: Only correct the existing analysis based on the identified issues.
        - **Professional Presentation**: Present the corrected analysis in a professional manner, formatted as a numbered list.

        Please proceed with the corrections.
        '''

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task_prompt}
    ]


def quality_check(analysis):
    # System message to set the context
    system_message = '''You are a grader looking to grade the a set of themes and their corresponding analysis 
    generated from a generative AI model.'''

    # Task prompt with reference to previous step
    task_prompt = f'''You previously identified the following themes with a corresponding explanation and Summarized examples: {analysis}

    For each theme, you were asked to provide the analysis in the following structure:
    
    1. **Key point**: A succinct key point (5-8 words).
    2. **Explanation**: A detailed explanation of the key point, continuing to focus on both commonalities and differences across universities.
    3. **Examples**: Provide a diverse set of examples, showing contrasting ways universities implement the policy. For example, "Universities X, Y, and Z provide full discretion for instructors to determine if students should use ChatGPT, while Universities A, B, and C strictly forbid its use under any circumstances."
    
    Your duty is to provide a grade to weather you feel that the analysis was satisfactory with respect to the requirements. 
    Examples of an unsatisfactory analysis includes: overlapping themes that could be easily merged into a common theme, vague explanations, 
    or "summarized examples" which do not provide a diverse set of examples, as demonstrated in the example provided. 
    Strictly return only "PASS" if you think that the analysis was satisfactory, or "FAIL" with an explanation otherwise. 
    '''
    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]


def quality_corrector(analysis, text, mistake):
    # System message to set the context
    system_message = '''You are a grader looking to grade the a set of themes and their corresponding analysis 
    generated from a generative AI model.'''

    # Task prompt with reference to previous step
    task_prompt = f'''You previously identified the following themes with a corresponding explanation and Summarized examples: {analysis}. 
    For each theme, you were asked to provide the analysis in the following structure:
    
    1. **Key Point**: A short, succinct statement (5-8 words) summarizing the theme. Example: "Privacy and security in generative AI use".
    2. **Explanation**: A detailed explanation of the key point, highlighting the commonalities in how universities address this issue.
    3. **Examples**: Some examples showing how different universities implement this policy. For example, "Universities A and B require students to document AI tool usage, while Universities X and Y prohibit AI assistance in assignments".

    
    A quality grader identified the following weaknesses in your analysis: {mistake}. 

    Your duty is to analyze the original documents and correct the mistake, preserving the same original analysis format.

    Here are the original policy documents: {text}
    '''
    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]




def get_common_themes(text):
    # System message to set the context
    system_message = '''You are a formal and professional assistant helping a higher education administrator 
    identify key themes from a summary of AI policies across top US universities. Your responses should be 
    accurate, neutral, and based solely on the provided information without any speculation or assumptions.'''

    # Task prompt for the model
    task_prompt = f'''Your task is to synthesize key themes from multiple AI policy summaries provided below. 
The policy summaries are in a dictionary format where each key is a university name, and each value is the corresponding summary of AI policies:

{text}

Please:

- Identify key themes associated with AI policies in higher education from the list of policy summaries. 
- Focus only on themes that are frequently mentioned across many institutions, and exclude any rare or unique policies found in only a few universities.
- **Rank the themes**, listing the most prominent (i.e., most frequently mentioned) themes first.
- Compare and consolidate similar policies into broader themes.

For each theme, include:

1. **Key Point**: A short, succinct statement (5-8 words) summarizing the theme. Example: "Privacy and security in generative AI use".
2. **Explanation**: A detailed explanation of the key point, highlighting the commonalities in how universities address this issue.
3. **Examples**: Examples showing how different universities implement this policy. For example, "Universities A and B require students to document AI tool usage, while Universities X and Y prohibit AI assistance in assignments".

**Instructions:**

- Base your response solely on the provided policy summaries.
- Do not introduce any new information or make assumptions.
- **Present the themes in a ranked, numbered list, starting with the most frequently mentioned theme**.
- Present the themes in a professional manner.

'''

    # Messages list for API call
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task_prompt}
    ]
    
    return messages


def refine_common_themes(text):
    # System message to set the context
    system_message = '''You are a formal and professional assistant helping a higher education administrator 
    refine key themes from a summary of AI policies across top US universities. Your responses should be 
    accurate, neutral, and free from speculation or assumptions.'''

    task_prompt = f'''You previously synthesized key points from multiple AI policy summaries from top US universities. 
Now, your task is to refine these themes by identifying those that share significant similarities or could be merged into overarching categories.

**Instructions:**

- **Group Similar Themes**: Carefully examine the provided themes. Group those that are similar or those that could be clustered into broader categories.
- **Preserve Original Ranking**: When presenting the grouped themes, maintain the original order of themes as much as possible, so that the most prominent themes remain at the top.

For each grouped theme, present it in the same format as provided:

1. **Key Point**: A short, succinct statement (5-8 words) summarizing the overarching theme. Example: "Privacy and security in generative AI use".
2. **Explanation**: A detailed explanation of the key point, highlighting the commonalities in how universities address this issue.
3. **Examples**: Examples showing how different universities implement this policy. For example, "Universities A and B require students to document AI tool usage, while Universities X and Y prohibit AI assistance in assignments".

**Additional Instructions:**

- Base your response solely on the provided themes and their content.
- Do not introduce new themes or external information.
- **Ensure that the original ranking is preserved**, so that the most prominent themes remain at the top of the list.
- Present the themes in a professional manner, formatted as a numbered list.

**Provided Themes:**

{text}
'''

    # Messages list for API call
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task_prompt}
    ]
    
    return messages



def generate_response_openAI(messages,api_key, model = "gpt-4o-mini", max_tokens = 5000):
    """
    Function to get a response from the OpenAI GPT-4 model.

    Parameters:
    prompt (str): The prompt that will be sent to the model for completion.
    api_key (str): The API key for the OpenAI API.
    model (str, optional): The model to be used for completion.
    Default to "gpt-4".
    max_tokens (int, optional): The maximum length of the generated text.
    Defaults to 1000.

    Returns:
    str: The generated text from the model.
    """
    client = OpenAI(
    api_key=api_key,
    )


    response = client.chat.completions.create(
        model = model,
        messages=messages,
        max_tokens=max_tokens, 
        temperature=0.1
    )
    return response.choices[0].message.content




def chain_prompts(text):
    print("CHAIN STARTING\n")
    themes=generate_response_openAI(get_common_themes((text)))
    final_set=generate_response_openAI(refine_common_themes(themes))

    grader, hallucinator=None, None


    print("-----"*10)
    print("CHECKING FOR QUALITY")
    while grader!="PASS":
        grader=generate_response_openAI(quality_check(final_set))
        if grader!="PASS":
            print(f"UNSATISFACTORY QUALITY: {grader}")
            final_set=generate_response_openAI(quality_corrector(final_set, text, grader))
    print("-----"*10)
    print("QUALITY CHECK PASSED!")


    print("-----"*10)
    print("CHECKING FOR HALLUCINATION")
    while hallucinator!="PASS":
        hallucinator=generate_response_openAI(hallucinator_check(final_set, text))
        if hallucinator!="PASS":
            print(f"HALLUCINATION IDENTIFIED: {hallucinator}")
            final_set=generate_response_openAI(hallucinator_corrector(final_set, text, hallucinator))
    print("-----"*10)
    print("HALLUCINATION CHECK PASSED!")

    return(final_set)




def save_to_file(data, file_name='../results/interim_results/key_points.txt'):
    with open(file_name, 'a') as file:
        file.write(data)  


def read_and_print_file(file_name='../results/interim_results/key_points.txt'):
    # Open the file in read mode
    with open(file_name, 'r') as file:
        data = file.read()  # Read the entire file content
    
    # Print the file content
    print(data)
    
    
