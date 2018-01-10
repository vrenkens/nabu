'''@file test_recipes.py
this function will test that all graphs in all recipe can actualy be built'''

import os
import test_recipe

def main():
    '''main function'''

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    test_dir(os.path.join('config', 'recipes'))

def test_dir(directory):
    '''test all recipes in this directory

    args:
        directory: the directory to test
    '''

    for recipe in os.listdir(directory):

        if not os.path.isdir(os.path.join(directory, recipe)):
            continue
        test_dir(os.path.join(directory, recipe))
        if is_recipe(os.path.join(directory, recipe)):
            test_recipe.main(os.path.join(directory, recipe))

def is_recipe(recipe):
    '''checks if directory is a recipe'''

    #the files that should definately be in a recipe
    files = ('database.conf', 'model.cfg', 'trainer.cfg')

    for f in files:
        if not os.path.exists(os.path.join(recipe, f)):
            return False

    return True

if __name__ == '__main__':
    main()
