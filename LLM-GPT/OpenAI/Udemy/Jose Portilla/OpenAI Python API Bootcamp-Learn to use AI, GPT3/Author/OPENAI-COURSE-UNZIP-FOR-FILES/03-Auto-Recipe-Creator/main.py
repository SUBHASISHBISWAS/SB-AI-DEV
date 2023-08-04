from recipe import RecipeGenerator
from dalle import Dish2Image

### Create an instance of the RecipeGenerator class ###
gen = RecipeGenerator()
###  Ask the user for input and create the recipe ###
recipe = gen.generate_recipe()
### Print the recipe ###
print(recipe)
### Create an instance of the Dish2Image class ###
dalle = Dish2Image(recipe)
### Visualize the dish ###
dalle.generate_image()
### Store the recipe in a text file ###
gen.store_recipe(recipe, f"{dalle.title}.txt")
