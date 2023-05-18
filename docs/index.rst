.. REPONAME documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to arxiv-astro-summarizer's documentation!
===============================
arxiv-astro-summarizer is an open-source program for scraping Astro-ph papers from the arXiv on some given date, summarizing the abstract, and comparing the similarity between the preprocessed abstract text and some specified user input. Can be used to scrape daily papers and filter for those relevant to some research topic.


Installation
==================
The current stable version can be installed via pip:

.. code-block:: bash

    pip install arxiv-astro-summarizer

Example
==================

.. code-block:: python
   
   from arxiv_astro_summarizer import astroph_summarizer
   
   #Create the class object
   scraper = astroph_summarizer.Scraper(date='2023-05-12', user_input='Black holes', path='/Users/daniel/Desktop/test')

   #Scrape papers from the input date
   scraper.scrape_arxiv()

   #Save all papers that were scraped
   scraper.save_paper(index='all')

   #Summarize the abstract of all papers and save the similary score
   scraper.summarize()

   #Remove the papers with similarity scores less than some threshold 
   scraper.remove_irrelevant_papers(similarity_threshold=0)

Pages
==================
.. toctree::
   :maxdepth: 1

   source/Page_1
   source/Page_2

Documentation
==================
Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/arxiv-astro-summarizer
