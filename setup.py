#!/usr/bin/env python
from os.path import join
import sys


if __name__ == '__main__':
    from setuptools import setup,find_packages

    try:
      from version import __version__
    except:
      __version__ = ''

    

    long_description = """
#####################################################################################
#                              easyastro V-0.1                                      #
#####################################################################################

Python source code dedicated to the Easy analysis of asttronomical data.
"""

    license = """	"""


    setup(name='easyastro',
      version='0.1',
        author='Samuel Gill',
        author_email='s.gill@keele.ac.uk',
        license='GNU GPLv3',
        url='https://github.com/samgill844/easyastro',
	packages=find_packages(),
    description="Python source code dedicated to the Easy analysis of asttronomical data.",
    long_description = long_description,
    classifiers = ['Development Status :: 4 - Beta'],
      install_requires=[
          'numpy', 'emcee', 'numba', 'astropy','astroplan','matplotlib'
      ]
	)   


'''
if (sys.version_info > (3, 0)):
    data_files=[ ('data/Gridss', ['data/Grids/spectrum.fits']), ( ], 
else:
'''
