from openai import OpenAI
import random
import numpy as np
from tqdm.notebook import tqdm_notebook
## check all these again
def generate_response_openAI(messages, max_tokens = 1000,api_key = "<insert openAI API key>", model = "gpt-4o-mini"):
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
    #messages=get_message(text)
    client = OpenAI(
    api_key=api_key,
    )


    response = client.chat.completions.create(
        model = model,
        messages=messages,
        max_tokens=max_tokens, 
        temperature=0
    )
    return response.choices[0].message.content



##################
def check_grade(analysis, query):
    system_message = '''You are a grader tasked with evaluating an analysis of synthesized policies derived from a set of AI policies stemming from various higher education institutes.'''
    
    task_prompt = f'''Based on the following query, you previously synthesized institutional summaries to identify some common distinct and non-overlapping policies 
    that universities adopt to address AI-related issues: "{query}"

    Your analysis identified these common policies: {analysis}

    Please provide a binary grade of "PASS" or "FAIL" based on the following three criteria: 
    1. Does each of the identified policies answer the core aspect as well as every specific detail of the query? 
    2. Are the identified policies relevant to the query?
    3. Are the identified policies distinct and non-overlapping as requested?
    4. Is the analysis detailed and sufficiently in-depth? Are the identified policies specific rather than being presented in a broad nature?

    If the analysis meets these criteria, respond strictly with only the word "PASS" and no additional text.  
    If it is unsatisfactory, respond with "FAIL" and specify which aspects require improvement.'''

    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]



def correct_grade(query, text, analysis, issue):
    system_message = '''You are an assistant tasked with improving an analysis that identifies common policies used by universities in addressing AI-related issues.'''
    
    task_prompt = f'''An analysis was previously conducted to identify distinct common policies used by universities 
    to address AI-related issues in higher education, based on the following query: "{query}"

    Original Analysis: {analysis}
    
    Feedback on issues with the analysis: {issue}
    
    Reference Documents: {text}
    
    Your task is to revise the analysis by addressing the feedback provided, ensuring all corrections are grounded in the original documents. 
    Maintain the original format of the analysis, and return only the revised analysis.'''

    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]

def quality_checker(text, query, analysis):
    grade,i=None, 0
    while grade!="PASS":
        if i>0 and i%2==0:
            print("TOO MANY ITERATIONS OF POOR QUALITY IDENTIFIED, REDO ANALYZING!")
            analysis=get_theme_analysis(text, query)
        else:
            pass


        fact_checker=generate_response_openAI(messages=check_grade(text, analysis))
        if (fact_checker.lower()).startswith("pass"):
            grade="PASS"
            print("QUALITY CHECK PASSED")
            break
        else:
            print(f"QUALITY CHECK FAILED: {fact_checker}")
            grade="FAIL"
            analysis=generate_response_openAI(messages=correct_grade(query, text, analysis, fact_checker))
        i+=1
    return(analysis)


####################
def check_hallucinations(text, analysis):
    system_message = '''You are a fact checker verifying if an analysis is grounded in the provided documents.'''
    
    task_prompt = f'''You previously analyzed a set of summarized AI policies across universities to identify common distinct and non-overlapping policies 
    employed to address AI-related issues within higher education.

    The required output format for each common policy was:
    - Policy: <The identified policy>  
    - Description: <a detailed description of the policy>
    - Examples: <Examples of institutions that use this policy>

    Here is the analysis to be verified:
    {analysis}

    Focus closely on whether the examples of institutions align with each institution's actual policy approach, based on the provided summaries.


    If the analysis is grounded, respond strictly with **only** the word "PASS" with no additional text.  
    If there are massive inaccuracies or fabrications, respond with "FAIL" and specify which aspects were hallucinated.
    
    - Define "hallucination" as any scenario where facts are massively fabricated or significantly inaccurate.
    - interpretive variations or paraphrasing nuances are not considered hallucinations and should strictly be a "PASS."
    - If analysis mostly grounded return a "PASS". 
 


    The set of summarized policies for each institution are in a dictionary format: {{"<University>: <summary of AI policies>"}}  
    Here set of summarized policies for each institution: {text}.'''

    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]

# - DO NOT give a "FAIL" simply because the analysis omits that does not endanger the overall factual correctness of the analysis.
def correct_hallucinations(text, analysis, issue):
    system_message = '''You are an assistant tasked with correcting information identified as hallucinated.'''
    
    task_prompt = f'''Previously, the following issues were identified in an analysis involving the identification of distinct and
    non-overlapping common policies used by universities to address AI-related issues in higher education:
    
    Issues Identified: {issue}
    
    Original Analysis: {analysis}
    
    Original Documents: {text}
    
    Your goal is to correct the identified issues in the analysis by grounding all information in the original documents provided. 
    Ensure that the revised analysis remains consistent with the original analysis format and is fully supported by the document contents.
    
    **Strictly return only the revised and corrected analysis.**'''

    return [{"role": "system", "content": system_message}, {"role": "user", "content": task_prompt}]

def fact_checker(text, query, analysis):
    grade,i=None, 0
    while grade!="PASS":
        if i>0 and i%2==0:
            print("TOO MANY ITERATIONS OF HALLUCINATION IDENTIFIED, REDOING ANALYSIS!")
            analysis=get_theme_analysis(text, query)
            analysis=quality_checker(text, query, analysis)
        else:
            pass


        fact_checker=generate_response_openAI(messages=check_hallucinations(text, analysis))
        if (fact_checker.lower()).startswith("pass"):
            grade="PASS"
            print("HALLUCINATION CHECK PASSED")
            break
        else:
            print(f"HALLUCINATION CHECK FAILED: {fact_checker}")
            grade="FAIL"
            analysis=generate_response_openAI(messages=correct_hallucinations(text, analysis, fact_checker))
        i+=1
    return(analysis)

def get_message(text, query):
    system_message = '''You are a formal assistant aiding a higher education administrator in analyzing and implementing policies 
    to address specific aspects of AI usage in higher education. Your responses should focus on identifying distinct and detailed
    policy employed by other universities. Present these policy in clear, bulleted key points, highlighting concise and practical approaches.'''

    task_prompt = f'''Based on the following query, you previously summarized AI policies across multiple universities: "{query}". 
    The policy summaries for each university are provided in a dictionary format, where each key is a university name, 
    and each value is the corresponding summary of its AI policies relevant to the query.

    Your task is to synthesize these institutional summaries and identify a concise set of **distinct and non-overlapping specific policies** commonly employed across
    universities to address AI usage related to the given query. 
    
    Here are the criteria: 
    1. Focus only on policies that are **frequently** observed across a majority of institutions and avoid those that appear infrequently.  
    2. Make sure each of the identified policies answers the core aspects and every specific detail of the given query. 
    3. Make sure each of the identified policies are highly relevant to the provided query. This means that I should be able to reconstruct the entire query simply by looking at any one of your identified policies. 
    4. Make sure the identified policies are specific in nature. Do not returning policies that are broad in nature. 
    5. Make sure the identified policies are strictly distinct and overlapping. 
    6. Order the identified policies from most common to least common. 
    7. Where possible, select policies that highlight **contrasting** approaches taken by different universities. 
    For example, for the query "How do universities handle the use of AI in assignments?," two contrasting policies could include:
    - "Some universities strictly prohibit the use of AI in assignments unless explicit permission is given by the instructor."
    - "Some universities allow the use of AI in assignments if it supports student learning."

    Format your response as follows:

    Policy: <The identified policy>  
    Description: <a detailed description of the policy>  
    Examples: <Examples of institutions that use this policy>  
    '''

    prompt = f"Synthesize common, specific, non-overlapping policies from the following summaries: {text}"
    
    message = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": task_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    return message





def get_theme_analysis(text, query):

    # Shuffle the list of items
    items=list(text.items())
    random.shuffle(items)
    shuffled_institute_retrievals = str(dict(items))
    summaries=generate_response_openAI(messages=get_message(text=shuffled_institute_retrievals, query=query))
    return(summaries)


def perform_chained_analysis(query, text):
    
    # Pre-allocate a numpy array with shape (5,) for 5 analyses (1-dimensional array of strings)
    num_analyses = 5
    analyses_array = np.empty(num_analyses, dtype=object)
    
    # Populate the array with text analysis results
    for i in tqdm_notebook(range(num_analyses)):
        initial_analysis = get_theme_analysis(text, query)
        graded_analysis=quality_checker(text, query, initial_analysis)
        fact_checked_analysis=fact_checker(text=text,query=query,analysis=graded_analysis)

        analyses_array[i]=fact_checked_analysis
    return analyses_array