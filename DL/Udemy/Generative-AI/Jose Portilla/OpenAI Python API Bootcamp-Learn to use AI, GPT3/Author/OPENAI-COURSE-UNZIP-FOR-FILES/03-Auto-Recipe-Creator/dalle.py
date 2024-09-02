import re
import requests
import shutil

import openai

class Dish2Image:

    def __init__(self, recipe):
        self.recipe = recipe
        self.title = Dish2Image._extract_title(recipe)

    def generate_image(self):
        prompt = self.dalle2_prompt()
        if Dish2Image._verify_prompt(prompt):
            response = Dish2Image.generate(prompt)
            image_url = response['data'][0]['url']
            return Dish2Image.save_image(image_url, f"{self.title}.jpg")
        raise ValueError("Prompt not accepted.")


    def dalle2_prompt(self):
        prompt = f"'{self.title}', professional food photography, 15mm, studio lighting"
        return prompt

    @staticmethod
    def _verify_prompt(prompt):
        print(prompt)
        response = input("Are you happy with the prompt? (y/n)")

        if response.upper() == "Y":
            return True
        return False


    @staticmethod
    def _extract_title(recipe):
        return re.findall("^.*Recipe Title: .*$", recipe, re.MULTILINE)[0].strip().split("Recipe Title: ")[1]

    @staticmethod
    def generate(image_prompt):
        response = openai.Image.create(prompt=image_prompt,
                                        n=1,
                                        size="1024x1024"
                                        )
        return response

    @staticmethod
    def save_image(image_url, file_name):
        image_res = requests.get(image_url, stream = True)
        
        if image_res.status_code == 200:
            with open(file_name,'wb') as f:
                shutil.copyfileobj(image_res.raw, f)
        else:
            print("Error downloading image!")
        return image_res.status_code

if __name__ == "__main__":
    """
    Test Dish2Image class based on the Recipe Title: Savory Tomato Apple Spaghetti
    """
    dalle = Dish2Image("Recipe Title: Savory Tomato Apple Spaghetti")
    dalle.generate_image()