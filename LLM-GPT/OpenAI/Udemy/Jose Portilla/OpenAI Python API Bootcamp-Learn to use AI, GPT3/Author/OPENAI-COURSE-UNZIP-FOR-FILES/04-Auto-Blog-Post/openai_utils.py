import os
import requests
import shutil

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_prompt(title):
    prompt = """Jose's Website
 
 Biography
 I am a Python instructor teaching people machine learning!


 Blog

 Jan 31, 2023
 Title: Why AI will never replace the radiologist
 tags: tech, machine-learning, radiology
 Summary:  I talk about the cons of machine learning in radiology. I explain why I think that AI will never replace the radiologist.
 Full text:""".format(title)
    return prompt

def get_blog_from_openai(blog_title):
    response = openai.Completion.create(engine="text-davinci-003",
                                            prompt=create_prompt(blog_title),
                                            max_tokens=512,  # we increased the tokens to get a longer blog post
                                            temperature=0.7)
    return response["choices"][0]["text"]


def dalle2_prompt(title):
    prompt = f"Pixel art showing '{title}'."
    return prompt


def save_image(image_url, file_name):
    image_res = requests.get(image_url, stream = True)
    
    if image_res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(image_res.raw, f)
    else:
        print("Error downloading image!")
    return image_res.status_code, file_name


def get_cover_image(title, save_path):
    response = openai.Image.create(
        prompt=dalle2_prompt(title),
        n=1,
        size="1024x1024"
        )
    image_url = response['data'][0]['url']
    status_code, file_name = save_image(image_url, save_path)
    return status_code, file_name


