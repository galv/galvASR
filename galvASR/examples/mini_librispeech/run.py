#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

from collections import OrderedDict

import click


@click.command()
@click.option('--stage', default=0)
def run(stage):
    for func_stage, func in stages.items():
        if func_stage < stage:
            pass
        else:
            func()


#@click.command()
#@click.argument(u'--data-dir')
#def download(data_dir):
#    pass



stages = OrderedDict(
    [(0, ), ]
)

if __name__ == u'__main__':
    run()
