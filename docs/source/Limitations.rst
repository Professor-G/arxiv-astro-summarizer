.. _Limitations:

Limitations
===========

Functionality
----------

By design this program queries scientific papers from the arXiv's Astrophysics (astro-ph) section! The `Scraper <https://arxiv-astro-summarizer.readthedocs.io/en/latest/autoapi/arxiv_astro_summarizer/astroph_summarizer/index.html#arxiv_astro_summarizer.astroph_summarizer.Scraper>`_ class can be used to query individual dates only, and the `scrape_and_analyze <https://arxiv-astro-summarizer.readthedocs.io/en/latest/autoapi/arxiv_astro_summarizer/astroph_summarizer/index.html#arxiv_astro_summarizer.astroph_summarizer.scrape_and_analyze>`_ function can be employed to query papers on a continous date range. Note that the output summary is intended solely for the purpose of calculating the similary metric, and is therefore not comprehensive. The purpose of this program is to identify papers related to some topic, and is thus not intended as a general purposes paper summarizer!

Date Range
----------

The scraping pipeline may return papers that were initally submitted to the arXiv outside the specified date range. This occurs when the arXiv performs bulk updates which alters the modification times of various submissions. Because of this bulk metadata record updates, the OAI-PMH datestamp values may not correspond with the original submission or replacement times for older articles, and may not for newer articles as well because of administrative and bibliographic updates!

As a workaround, the program will inspect the top lines of the first page in an attempt to locate the date -- if no date is present at the top of the page, it checks the left hand-side where the arXiv record is printed (the vertical text). If the input date is not present in either of these two locations, the paper is removed from the working directory. This procedure eliminates the majority of the false-positives, but given the variety of article page formats, it's possible that no date is present or can be extracted, and thus papers outside your input date range may be saved to your working directory.

Identical Author Names
----------

If two separate papers have a first author with the same last name, and one of these papers falls outside the date range as per the aforementioned issue, both papers will ultimately end up being removed. This is because each paper is saved in the following format: first_author_last_name_YYYY. 

Future program development will address this issue. 






