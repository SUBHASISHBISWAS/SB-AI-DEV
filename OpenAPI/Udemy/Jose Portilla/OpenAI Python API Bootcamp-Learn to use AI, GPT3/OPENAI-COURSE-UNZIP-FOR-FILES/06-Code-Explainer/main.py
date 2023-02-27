import inspect  # Used to transform python code to a string
import os
from pathlib import Path

import openai
import functions

openai.api_key = os.getenv("OPENAI_API_KEY")

def docstring_prompt(code):
    prompt = f"{code}\n # A high quality python docstring of the above python function:\n\"\"\""
    return prompt

def merge_docstring_and_function(function_string, docstring):
    split = function_string.split("\n")
    first_part, second_part = split[0], split[1:]
    merged_function = first_part + "\n" + '    """' + docstring + '    """' + "\n" + "\n".join(second_part)
    return merged_function

### Extract all functions from the functions.py file ###
### getmembers returns a tuple, where the first element is the name of the function and the second element is the function itself ###
### We only want the functions that are defined in the functions.py file, so we filter out the functions that are defined in other modules ###

def get_all_functions(module):
    return [mem for mem in inspect.getmembers(module, inspect.isfunction)
         if mem[1].__module__ == module.__name__]


def get_openai_completion(prompt):
    response = openai.Completion.create(
                model="code-davinci-002",
                prompt=prompt,
                temperature=0,
                max_tokens=256,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\"\"\""] # Corresponds to """, the end of the docstring
                )
    return  response["choices"][0]["text"]



### Create a prompt for each function and ###
### Run the prompts through the model and store the results ###
if __name__ == "__main__":
    functions_to_prompt = functions

    all_funcs = get_all_functions(functions)

    functions_with_prompts = []
    for func in all_funcs:

        code = inspect.getsource(func[1])
        prompt = docstring_prompt(code)
        response = get_openai_completion(prompt)

        
        merged_code = merge_docstring_and_function(code, response)
        functions_with_prompts.append(merged_code)
        

    functions_to_prompt_name = Path(functions_to_prompt.__file__).stem
    with open(f"{functions_to_prompt_name}_withdocstring.py", "w") as f:
        f.write("\n\n".join(functions_with_prompts))

