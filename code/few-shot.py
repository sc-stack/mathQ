# Importing necessary libraries
import openai  # Library for interacting with the OpenAI API
import pandas as pd  # Library for handling data in tabular form
import time  # Library for time-related functions
import argparse  # Library for handling command-line arguments
import os  # Library for interacting with the operating system

# Retrieving the OpenAI API key from environment variables for security
openai.api_key = os.getenv('OpenAI_API_Key')

# List of course codes and math sections to be processed
courses_to_few_shot = ['18.01', '18.02', '18.03', '6.042', '18.05', '18.06', 'COMS3251']
MATH_sections_to_few_shot = [
    'MATH_Algebra', 'MATH_Counting_&_Probability', 'MATH_Intermediate_Algebra', 
    'MATH_Number_Theory', 'MATH_Prealgebra', 'MATH_Precalculus'
]
questions_per_course = 25  # Number of questions to process per course
questions_per_MATH_section = 15  # Number of questions to process per math section

# Setting up the argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--Codex_Few_Shot")
parser.add_argument("--GPT3_CoT_One_Shot")
parser.add_argument("--Do_MATH")
parser.add_argument("--Do_Courses")
args = parser.parse_args()

# Configuration settings for the API calls and processing
few_shot_examples_desired = 5  # Desired number of few-shot examples to use
codex_engine = "code-davinci-002"  # The OpenAI model for processing code-related queries
gpt3_engine = "text-davinci-002"  # The OpenAI model for processing natural language queries
engine_temperature = 0  # The randomness of the output (0 for deterministic)
engine_topP = 0  # The likelihood of the most probable tokens being chosen (0 for deterministic)
few_shot_max_tokens = 256  # Maximum number of tokens in the output for few-shot queries
gpt3_CoT_max_tokens = 1000  # Maximum number of tokens in the output for GPT-3 queries
codex_time_delay = 3  # Delay in seconds between Codex API calls to avoid rate limiting
gpt3_time_delay = 1  # Delay in seconds between GPT-3 API calls to avoid rate limiting
CoT = "Let's think step by step."  # A phrase to initiate a chain of thought process in the model

# Function to process few-shot learning for the provided courses
def execute_few_shot(courses, questions_per):
    """
    Runs few-shot on questions_per questions for each course in courses.
    """
    # Processing each course
    for course in courses:
        course_location = course + ' results.csv'  # Location of the results file for the course
        # Initializing new columns in the CSV to store few-shot results
        results = pd.read_csv(course_location)
        results['Few-Shot Input'] = ''
        results['Few-Shot Output'] = ''
        results['Few-Shot Evaluation'] = ''
        results.to_csv(course_location, index=False)

        # Processing each question
        for i in range(questions_per):
            k = few_shot_examples_desired  # Counter for the desired number of few-shot examples

            # If the question was already correct in zero-shot, no need for few-shot
            if results.iloc[i]['Zero-Shot Evaluation'] == 1:
                print('no few shot needed for ' + course + ' question ' + str(i+1))
                few_shot_input = 'n/a'
                few_shot_output = 'n/a'

            # If the question was incorrect in zero-shot, perform few-shot
            elif results.iloc[i]['Zero-Shot Evaluation'] == 0:
                few_shot_input = ''
                print('doing few-shot for ' + course + ' question ' + str(i+1) + '...')
                # Building the few-shot prompt with examples of similar questions that were correct
                for closest in results.iloc[i]["Most Similar Questions"].strip('][').split(', '):
                    closest_index = int(closest) - 1
                    if results.iloc[closest_index]['Zero-Shot Evaluation'] == 1 and k > 0:
                        few_shot_input += results.iloc[closest_index]['Codex Input']
                        few_shot_input += results.iloc[closest_index]['Codex Output']+'\n\n'
                        k -= 1
                few_shot_input += results.iloc[i]['Codex Input']
                start = time.time()
                time.sleep(codex_time_delay)  # Delay to avoid rate limit errors
                # Making an API call to OpenAI Codex with the few-shot prompt
                response = openai.Completion.create(
                    engine=codex_engine,
                    prompt=few_shot_input,
                    temperature=engine_temperature,
                    max_tokens=few_shot_max_tokens,
                    top_p=engine_topP
                )
                few_shot_output = response.choices[0].text.strip()
                end = time.time()
                print('few-shot for ' + course + ' question ' + str(i+1) + ' took ' + str(end - start) + ' seconds')
                # Evaluating the response
                if few_shot_output == results.iloc[i]['Solution']:
                    few_shot_evaluation = 1
                else:
                    few_shot_evaluation = 0
            # Updating the results CSV with the few-shot results
            results.iloc[i, results.columns.get_loc('Few-Shot Input')] = few_shot_input
            results.iloc[i, results.columns.get_loc('Few-Shot Output')] = few_shot_output
            results.iloc[i, results.columns.get_loc('Few-Shot Evaluation')] = few_shot_evaluation
            results.to_csv(course_location, index=False)

# Function to process the chain of thought (CoT) method for GPT-3
def execute_gpt3_CoT(sections, questions_per):
    """
    Runs the GPT-3 chain of thought method on questions_per questions for each section in sections.
    """
    # Processing each section
    for section in sections:
        section_location = section + ' results.csv'  # Location of the results file for the section
        # Initializing new columns in the CSV to store CoT results
        results = pd.read_csv(section_location)
        results['GPT-3 CoT Input'] = ''
        results['GPT-3 CoT Output'] = ''
        results['GPT-3 CoT Evaluation'] = ''
        results.to_csv(section_location, index=False)

        # Processing each question
        for i in range(questions_per):
            # If the question was already correct in zero-shot, no need for CoT
            if results.iloc[i]['Zero-Shot Evaluation'] == 1:
                print('no CoT needed for ' + section + ' question ' + str(i+1))
                gpt3_CoT_input = 'n/a'
                gpt3_CoT_output = 'n/a'

            # If the question was incorrect in zero-shot, perform CoT
            elif results.iloc[i]['Zero-Shot Evaluation'] == 0:
                gpt3_CoT_input = CoT + results.iloc[i]['GPT-3 Input']
                print('doing GPT-3 CoT for ' + section + ' question ' + str(i+1) + '...')
                start = time.time()
                time.sleep(gpt3_time_delay)  # Delay to avoid rate limit errors
                # Making an API call to OpenAI GPT-3 with the CoT prompt
                response = openai.Completion.create(
                    engine=gpt3_engine,
                    prompt=gpt3_CoT_input,
                    temperature=engine_temperature,
                    max_tokens=gpt3_CoT_max_tokens,
                    top_p=engine_topP
                )
                gpt3_CoT_output = response.choices[0].text.strip()
                end = time.time()
                print('GPT-3 CoT for ' + section + ' question ' + str(i+1) + ' took ' + str(end - start) + ' seconds')
                # Evaluating the response
                if gpt3_CoT_output == results.iloc[i]['Solution']:
                    gpt3_CoT_evaluation = 1
                else:
                    gpt3_CoT_evaluation = 0
            # Updating the results CSV with the CoT results
            results.iloc[i, results.columns.get_loc('GPT-3 CoT Input')] = gpt3_CoT_input
            results.iloc[i, results.columns.get_loc('GPT-3 CoT Output')] = gpt3_CoT_output
            results.iloc[i, results.columns.get_loc('GPT-3 CoT Evaluation')] = gpt3_CoT_evaluation
            results.to_csv(section_location, index=False)

# Main function to run the appropriate methods based on command-line arguments
if __name__ == "__main__":
    if args.Do_MATH:
        execute_gpt3_CoT(MATH_sections_to_few_shot, questions_per_MATH_section)
    if args.Do_Courses:
        execute_few_shot(courses_to_few_shot, questions_per_course)
