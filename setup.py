# -*- coding: utf-8 -*-
"""
Created on Fri June 6 3 16:40:44 2022

@author: danielgodinez
"""
from setuptools import setup, find_packages, Extension


setup(
    name="arxiv-astro-summarizer",
    version="0.11",
    author="Daniel Godines",
    author_email="danielgodinez123@gmail.com",
    description="Scrapes arXiv astro-ph paper, summarizes the abstract, and returns relavant papers according to a user input.",
    license='GPL-3.0',
    url = "https://github.com/Professor-G/arxiv-astro-summarizer",
    classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3',	   
],
    packages=find_packages('.'),
    install_requires = ['numpy', 'requests', 'arxivscraper', 'pandas', 'PyPDF2', 'transformers', 'scikit-learn', 'nltk', 'textract', 'datefinder'],
    python_requires='>=3.7,<4',
    include_package_data=True,
    test_suite="nose.collector",
)
