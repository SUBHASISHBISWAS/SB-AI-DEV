from pathlib import Path
import blog_utils
import openai_utils
import os

### Define your paths here

PATH_TO_BLOG_REPO = Path('/Users/subhasish/Development/GIT/Interstellar/subhasishbiswas.github.io/.git')
PATH_TO_BLOG = PATH_TO_BLOG_REPO.parent
PATH_TO_CONTENT = PATH_TO_BLOG/"content"
PATH_TO_CONTENT.mkdir(exist_ok=True, parents=True)
###

# Define a title and get the blog content from OpenAI
title = "Why We Need AI"
print(openai_utils.create_prompt(title))
blog_content = openai_utils.get_blog_from_openai(title)

# Get the cover image and create the blog
PATH_TO_SAVE_IMAGE_PARENT=Path(os.getcwd()).parent
PATH_TO_SAVE_IMAGE=PATH_TO_SAVE_IMAGE_PARENT/"Images"
PATH_TO_SAVE_IMAGE.mkdir(exist_ok=True, parents=True)
print(f"PATH_TO_SAVE_IMAGE ::{PATH_TO_SAVE_IMAGE}")
_, cover_image_save_path = openai_utils.get_cover_image(title, f"{PATH_TO_SAVE_IMAGE}/cover_image.jpg")
path_to_new_content = blog_utils.create_new_blog(PATH_TO_CONTENT, title, blog_content, cover_image_save_path)
blog_utils.write_to_index(PATH_TO_BLOG, path_to_new_content)

# Update the blog
blog_utils.update_blog(PATH_TO_BLOG_REPO)