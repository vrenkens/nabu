'''@file test_recipe.py
test if a recipe can build the graphs correctly'''

import sys
import os
import traceback
import train
import test
import decode

def main(recipe):
    '''main function

    args:
        recipe: the recipe to be tested
    '''

    print 'testing %s' % recipe
    sys.stderr = open(os.devnull, 'w')

    #test training
    try:
        sys.stdout = open(os.devnull, 'w')
        train.train(
            clusterfile=None,
            job_name='local',
            task_index=0,
            ssh_command='None',
            expdir=recipe,
            testing=True
        )
        decode.decode(recipe, True)
        test.test(recipe, True)
        sys.stdout = sys.__stdout__

    #pylint: disable=W0702
    except:
        sys.stdout = sys.__stdout__
        print 'error for recipe %s:' % recipe
        print traceback.format_exc()
        return

    print '%s OK' % recipe

if __name__ == '__main__':
    main(sys.argv[1])
