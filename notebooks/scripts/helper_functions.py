"""
Copyright 2024 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import json
from langchain.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
import logging
import os
from timeit import default_timer as timer

# Create the logger
DEFAULT_LOG_LEVEL = logging.NOTSET
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
log_level = os.environ.get('LOG_LEVEL')
match log_level:
    case '10':
        log_level = logging.DEBUG
    case '20':
        log_level = logging.INFO
    case '30':
        log_level = logging.WARNING
    case '40':
        log_level = logging.ERROR
    case '50':
        log_level = logging.CRITICAL
    case _:
        log_level = DEFAULT_LOG_LEVEL
log_format = os.environ.get('LOG_FORMAT')
if log_format is None:
    log_format = DEFAULT_LOG_FORMAT
elif len(log_format) == 0:
    log_format = DEFAULT_LOG_FORMAT
# Set the basic config for the lgger
logging.basicConfig(level=log_level, format=log_format)


# Function to get counts from text
def get_counts_from_text(text):
    if text is None:
        text = ''
    char_count = len(text)
    word_count = len(text.split())
    return char_count, word_count


# Function to check if the specified modality exists in both input and output
def does_modality_exists(input_modality_list, output_modality_list, required_modality):
    if required_modality in input_modality_list:
        if required_modality in output_modality_list:
            return True
        else:
            return False
    else:
        return False


# Function to get the max output token length for the specified model
def get_max_output_length(model_id):
    # These limits have been obtained from
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
    # For Anthropic models, the provider recommends a limit of 4000
    # for optimal performance even though they support 4096
    model_id_output_length_dict = {
        'amazon.titan-text-lite-v1': 8000,
        'amazon.titan-text-express-v1': 8000,
        'anthropic.claude-instant-v1': 4000,
        'anthropic.claude-v2': 4000,
        'anthropic.claude-v2:1': 4000,
        'ai21.j2-mid-v1': 8191,
        'ai21.j2-ultra-v1': 8191,
        'cohere.command-light-text-v14': 4096,
        'cohere.command-text-v14': 4096,
        'meta.llama2-13b-chat-v1': 2048,
        'meta.llama2-70b-chat-v1': 2048,
        'mistral.mistral-7b-instruct-v0:2': 8192,
        'mistral.mixtral-8x7b-instruct-v0:1': 4096
    }
    return model_id_output_length_dict.get(model_id, 0)


# Function to create the model-specific inference parameters
def get_model_kwargs(model_id, temperature, max_response_token_length):
    # Check and substitute for the model's max response token length if it is specified as '-1'
    if max_response_token_length == -1:
        max_response_token_length = get_max_output_length(model_id)
    # Generate the model-specific inference parameters
    match model_id:
        case 'amazon.titan-text-lite-v1' | 'amazon.titan-text-express-v1':
            model_kwargs = {
                "temperature": temperature,
                "maxTokenCount": max_response_token_length
            }
        case 'anthropic.claude-instant-v1' | 'anthropic.claude-v2' | 'anthropic.claude-v2:1':
            model_kwargs = {
                "temperature": temperature,
                "max_tokens_to_sample": max_response_token_length
            }
        case 'ai21.j2-mid-v1' | 'ai21.j2-ultra-v1':
            model_kwargs = {
                "temperature": temperature,
                "maxTokens": max_response_token_length
            }
        case 'cohere.command-light-text-v14' | 'cohere.command-text-v14':
            model_kwargs = {
                "temperature": temperature,
                "max_tokens": max_response_token_length
            }
        case 'meta.llama2-13b-chat-v1' | 'meta.llama2-70b-chat-v1':
            model_kwargs = {
                "temperature": temperature,
                "max_gen_len": max_response_token_length
            }
        case 'mistral.mistral-7b-instruct-v0:2' | 'mistral.mixtral-8x7b-instruct-v0:1':
            model_kwargs = {
                "temperature": temperature,
                "max_tokens": max_response_token_length
            }
        case _:
            model_kwargs = None
    # Return the model kwargs
    return model_kwargs
    

# Function to read the content of the specified file
def read_file(dir_name, file_name):
    logging.info('Reading content from file "{}"...'.format(file_name))
    with open(os.path.join(dir_name, file_name)) as f:
        content = f.read()
    logging.info('Completed reading content from file.')
    return content


# Function to prepare the prompt
def prepare_prompt(prompt_template_dir, prompt_template_file_name, **kwargs):
    prompt_template_file_path = os.path.join(prompt_template_dir, prompt_template_file_name)
    logging.info('Reading content from prompt template file "{}"...'.format(prompt_template_file_name))
    prompt_template = PromptTemplate.from_file(prompt_template_file_path)
    logging.info('Completed reading content from prompt template file.')
    logging.info('Substituting prompt variables...')
    prompt = prompt_template.format(**kwargs)
    logging.info('Completed substituting prompt variables.')
    return prompt


# Function to invoke the specified LLM through the boto3 Bedrock Runtime
# client and using the specified prompt
def invoke_llm(llm, prompt):
    logging.info('Invoking LLM "{}" with specified inference parameters "{}"...'.
                 format(llm.model_id, llm.model_kwargs))
    start = timer()
    prompt_response = llm.invoke(prompt)
    end = timer()
    logging.info(prompt + prompt_response)
    logging.info('Completed invoking LLM.')
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return prompt_response


# Function to invoke the specified LLM through the Bedrock Runtime client and
# using the specified prompt
def invoke_llm_with_bedrock_rt(model_id, bedrock_rt_client, temperature, max_response_token_length, prompt):
    # Create the request body
    json_body = get_model_kwargs(model_id, temperature, max_response_token_length)
    model_kwargs = json.dumps(json_body)
    json_body.update({"prompt": prompt})
    body = json.dumps(json_body)
    logging.info('Invoking LLM "{}" with specified inference parameters "{}"...'.
                 format(model_id, model_kwargs))
    start = timer()
    invoke_model_response = bedrock_rt_client.invoke_model(
        body=body,
        modelId=model_id
    )
    end = timer()
    logging.info('Completed invoking LLM.')
    # Parse the response body
    response_body = json.loads(invoke_model_response.get('body').read())
    outputs = response_body.get('outputs')
    prompt_response = outputs[0].get('text')
    logging.info(prompt + prompt_response)
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return prompt_response


# Function to invoke the specified LLM through the LangChain client and
# using the specified prompt
def invoke_llm_with_lc(model_id, bedrock_rt_client, temperature, max_response_token_length, prompt):
    # Create the LangChain LLM client
    logging.info('Creating LangChain client for LLM "{}"...'.format(model_id))
    llm = Bedrock(
        model_id = model_id,
        model_kwargs = get_model_kwargs(model_id, temperature, max_response_token_length),
        client = bedrock_rt_client
    )
    logging.info('Completed creating LangChain client for LLM.')
    logging.info('Invoking LLM "{}" with specified inference parameters "{}"...'.
                 format(llm.model_id, llm.model_kwargs))
    start = timer()
    prompt_response = llm.invoke(prompt)
    end = timer()
    logging.info(prompt + prompt_response)
    logging.info('Completed invoking LLM.')
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return prompt_response


# Function to process the steps required in the example prompt 1
def process_prompt_1(model_id, bedrock_rt_client, temperature, max_response_token_length,
                     prompt_templates_dir, prompt_template_file, prompt_data, call_to_action):
    # Read the prompt template and perform variable substitution
    prompt = prepare_prompt(prompt_templates_dir, prompt_template_file,
                            DATA=prompt_data, CALL_TO_ACTION=call_to_action)
    # Invoke the LLM and print the response
    match model_id:
        case 'mistral.mistral-7b-instruct-v0:2' | 'mistral.mixtral-8x7b-instruct-v0:1':
            return invoke_llm_with_bedrock_rt(model_id, bedrock_rt_client, temperature,
                                              max_response_token_length, prompt)
        case _:
            return invoke_llm_with_lc(model_id, bedrock_rt_client, temperature,
                                      max_response_token_length, prompt)