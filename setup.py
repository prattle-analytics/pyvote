

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyvote',
    version='0.0.0',

    description='allows for model voting for binary and categorical classifiers of mixed types (keras, sklearn)',
    long_description=long_description,
    url='https://github.com/prattle-analytics/pyvote',
    author='Prattle Analytics ',
    author_email='support@prattle.co',
    license='public',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='classification, neural nets, voting, ensemble models',

    py_modules=["pyvote"],
)
