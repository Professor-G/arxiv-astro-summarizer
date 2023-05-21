.. arxiv-astro-summarizer documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to arxiv-astro-summarizer's documentation!
===============================
This is an open-source program for scraping Astro-ph papers from the arXiv on some given date, summarizing the abstract, and comparing the preprocessed abstract text to some specified user input, from which a similarity score is computed. This program can be used to scrape daily papers and filter for those relevant to some research topic(s).

Installation
------------
The current stable version can be installed via pip:

.. code-block:: bash

    pip install arxiv-astro-summarizer

Example: Single Date
------------

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

   #Print the dataframe
   scraper.df 
   
   #Remove the papers with similarity scores less than some threshold 
   scraper.remove_irrelevant_papers(similarity_threshold=0)

Example: Date Range
------------

Given a continuous date range, you can use the ``scrape_and_analyze`` function, which will loop through the dates to automatically perform the above steps for you:

.. code-block:: python

   start_date = '2023-04-01'
   end_date = '2023-04-30'
   user_input = 'Black holes'
   similarity_threshold = 0.1

   astroph_summarizer.scrape_and_analyze(start_date, end_date, user_input=user_input, similarity_threshold=similarity_threshold, path=None)

Note that the ``similarity_threshold`` parameter can be set to ``None``, in which case all papers within the date range will be saved.

Important
------------

This program employs the use of word lemmatizers to improve text summarization by transforming words into their base forms and capturing alternative words or synonyms. This integration enhances the program's ability to generate concise and informative summaries that accurately reflect the original text. Nonetheless, as with all large language models, the more specific the ``user_input`` is, the more accurate the calculated similarity score will be -- as such, it is advised to keep the input as concise as possible and to avoid the use of stop words.

Pages
==================
.. toctree::
   :maxdepth: 1

   source/Limitations

Documentation
==================
Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/arxiv-astro-summarizer
