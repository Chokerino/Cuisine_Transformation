import json
import io
import pandas as pd

with open("recipe_with_cuisine.json",encoding="utf8") as file:
    data=json.load(file)
    print(data)