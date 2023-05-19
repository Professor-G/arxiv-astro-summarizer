#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:10:00 2023

@author: daniel
"""
import re
import logging
import requests
import unicodedata
import textract
import datefinder
from progress import bar

import arxivscraper
import numpy as np 
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
logging.getLogger("transformers").setLevel(logging.ERROR)
import os; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from textract.exceptions import MissingFileError

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt') 
nltk.download('stopwords'); nltk.download('stopwords'); nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Scraper:
    """
    Class object to scrape, read, and analyze arXiv papers published on a given date.

    Args:
        date (str): The initial date to consider for scraping, in the following format: YYYY-MM-DD, e.g. '2022-05-12'.
        user_input (str, optional): A short excerpt describing the kind of papers the user is interested in. Defaults to None.
        path (str, optional): The path where the file should be saved. Defaults to None, which saves the files to the local home directory.

    Attributes:
        df (pandas.DataFrame): DataFrame containing the metadata of the scraped arXiv papers.
        text (str): Processed abstract text extracted from a PDF file.
        raw_text (str): Raw abstract text extracted from a PDF file.
        filenames (list): List of saved file names.

    Raises:
        ValueError: If the input date is not a string or if user_input is not a string when provided.

    Examples:
        Initialize the Scraper object with a date and user input:
        >>> scraper = Scraper('2022-05-12', user_input='black hole')

        Scrape arXiv papers for the given date:
        >>> scraper.scrape_arxiv()

        Save a specific paper using its index in the scraped metadata, or set to 'all' to save all papers:
        >>> scraper.save_paper(0)

        Summarize the saved papers and score the relevance between the summary and the user_input:
        >>> scraper.summarize()

        Remove papers with similarity scores less than or equal to the specified similarity_threshold (defaults to 0)
        >>> scraper.remove_irrelevant_papers(similarity_threshold=0)
    """

    def __init__(self, date, user_input=None, path=None):
        self.date = date
        self.user_input = user_input
        self.path = path
        
        self.df = None
        self.text = None 
        self.raw_text = None 
        self.filenames = [] 

        self.path = str(Path.home()) if self.path is None else self.path
        self.path += '/' if self.path[-1] != '/' else ''

        if not isinstance(self.date, str):
            raise ValueError('The input date should be a string in the format MM DD, YYYY or MM DD YYYY')

        if user_input is not None and not isinstance(self.user_input, str):
            raise ValueError('The user_input must be in string format!')

        print(f"Files will be saved in: {self.path}")

    def format_date(self):
        """
        Converts a date string from 'YYYY-MM-DD' format to 'DD Month YYYY' format.
        This will be used to keyword search the paper, to ensure that the publication 
        times are consistent with the input date.

        Returns:
            str: The date string in 'DD Month YYYY' format.
        """
        
        formatted_date = datetime.strptime(self.date, '%Y-%m-%d')
        formatted_date = formatted_date.strftime('%d %B %Y')
        
        return formatted_date

    def check_pdf_contains_text(self, filename, text):
        """
        Check if a PDF file contains a specific text.

        Args:
            filename (str): The path to the PDF file.
            text (str): The text to search for in the PDF.

        Returns:
            bool: True if the text is found in the PDF, False otherwise.
        """

        def clean_text(text):
            """
            Clean the text by removing unwanted characters and excessive whitespace.

            Args:
                text (str): The text to clean.

            Returns:
                str: The cleaned text.
            """

            cleaned_text = re.sub(r'[^\w\s-]', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'(\w+)-(\w+)', r'\1 - \2', cleaned_text)

            return cleaned_text.strip()

        def extract_dates_from_text(text):
            """
            Extract dates from the given text.

            Args:
                text (str): The text to extract dates from.

            Returns:
                list: A list of extracted dates in the format '%d %b %Y'.
            """

            dates = []
            matches = datefinder.find_dates(text, strict=True)
            for match in matches:
                date_str = match.strftime('%d %b %Y')
                dates.append(date_str)

            return dates

        with open(filename, 'rb') as file:
            pdf_reader = PdfReader(file)
            first_page = pdf_reader.pages[0]
            page_text = first_page.extract_text()

            # Check for the text in the left-hand side
            left_text = page_text.split('\n')[-1:]
            #print('=========================');print('=========================');print('filename', filename);print('text', text);print('left_text', left_text)
            if text in left_text[0]:
                return True

            # Check for the text in the top ten lines
            top_text = ' '.join(page_text.split('\n')[:10])
            cleaned_top_text = clean_text(top_text)
            dates = extract_dates_from_text(cleaned_top_text)
            #print('top_text', top_text);print('cleaned_top_text', cleaned_top_text);print('Extracted Dates:', dates);print('=========================')
            if text in dates:
                return True

        return False

    def return_arxiv_url(self, doi):
        """
        Returns the https url of an astro-ph paper, given
        by the following hardcoded format: https://arxiv.org/pdf/{doi}
        as such, do not include anything other than the numbers!

        Returns:
            Returns the path to the arxiv pdf (without the pdf extension)
        """
      
        full_path = 'https://arxiv.org/pdf/' + str(doi)

        return full_path

    def scrape_arxiv(self):
        """
        Compiles the metadata of the arxiv papers uploaded on a given day. Scraping
        date ranges is not currently supported!
    
        Returns:
            Pandas dataframe with the following meta-data: 'id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors'
        """
        
        if not isinstance(self.date, str):
            raise ValueError('date should be a string in the format YYYY-MM-DD')

        scraper = arxivscraper.Scraper(category='physics:astro-ph', date_from=self.date, date_until=self.date)
        output = scraper.scrape()
        
        self.df = pd.DataFrame(output, columns=('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors'))
        
        return

    def save_paper(self, index):
        """
        Function to download and save a pdf given the DOI number. The file 
        is saved to the specified directory, or the local home directory if path is not provided.
        
        Args:
            index (int or str): Index of the paper in the meta-data, which is saved as an attribute
                called ``df`` after running the scrape_arxiv() class method. Can also be set to 'all', in
                which case the entire DataFrame will be looped through to save all appropriate papers.
            
        Returns:
            No return, the file is either saved to the path or it's not.
        """
        
        def format_filename(filename):
            """
            Formats the given filename by capitalizing each part separated by hyphens,
            converting it to title case, and adding the '.pdf' extension.

            Args:
                filename (str): The filename to be formatted.

            Returns:
                str: The formatted filename.
            """

            filename_parts = filename.split('-')
            filename_parts = [part.capitalize() for part in filename_parts]
            filename = '-'.join(filename_parts)
            filename = filename.title()
            filename += '.pdf'

            return filename

        if self.df is None:
            raise ValueError('Meta-data has not been scraped, run the scrape_arxiv() class method first!')
           
        if index == 'all':

            progess_bar = bar.FillingSquaresBar('Saving files......', max=len(self.df))
            
            for i in range(len(self.df)):

                filename = format_filename(self.df.iloc[i].authors[0].split()[-1] + '_' + self.df.iloc[i].created[:4]) # Filename, author_YYYY

                try:
                    response = requests.get(self.return_arxiv_url(self.df.id.iloc[i]))

                    with open(self.path + filename, 'wb') as f:
                        f.write(response.content)

                    if self.check_pdf_contains_text(self.path + filename, self.format_date()):
                        #print('File saved in: {}'.format(self.path + filename))
                        self.filenames.append(filename)
                    else:
                        os.remove(self.path + filename)
                        #print(f"{filename[:-4]} outside date range, removing...")
                except Exception as e:
                    print('An error occurred while downloading the PDF: {}'.format(str(e)))

                progess_bar.next()

            progess_bar.finish()

        else:

            filename = format_filename(self.df.iloc[index].authors[0].split()[-1] + '_' + self.df.iloc[index].created[:4])  # Filename, author_YYYY

            try:
                response = requests.get(self.return_arxiv_url(self.df.id.iloc[index]))

                with open(self.path + filename, 'wb') as f:
                    f.write(response.content)

                if self.check_pdf_contains_text(self.path + filename, self.format_date()):
                    #print('File saved in: {}'.format(self.path + filename))
                    self.filenames.append(filename)
                else:
                    os.remove(self.path + filename)
                    #print(f"{filename[:-4]} outside date range, removing...")
            except Exception as e:
                print('An error occurred while downloading the PDF: {}'.format(str(e)))

        return

    def extract_abstract_from_pdf(self, filename):
        """
        Extracts the abstract from a PDF file.

        Args:
            filename (str): The path to the PDF file.

        Returns:
            str: The extracted abstract text.
        """

        self.raw_text = textract.process(filename, method='pdfminer').decode('utf-8')

        # Check if the word "abstract" is present
        match_abstract = re.search(r'Abstract([\s\S]+?)(?:Key\s*words|$)', self.raw_text, re.IGNORECASE)
        if match_abstract:
            self.raw_text = match_abstract.group(1).strip()
        else:
            # Extract the paragraph after authors and before the first section
            match_abstract = re.search(r'(?:\n\n)(?!.*Abstract)(?![A-Z]{2,})(?![0-9]+\.)[\s\S]+?(?=\n\n[A-Z]{2,}|\n\n[0-9]+\.)', self.raw_text)
            if match_abstract:
                self.raw_text = match_abstract.group(0).strip()[:2000] #Keep only the first 2000 characters of the first page 
            else:
                self.raw_text = ''

        return self.raw_text
        
    def process_abstract(self, replace_astro_terms=True):
        """
        Preprocesses the abstract text by applying specific transformations.

        Args:
            text (str): The input abstract text to be preprocessed.

        Returns:
            str: The preprocessed abstract text.
        """

        if self.raw_text is None:
            raise ValueError('Text has not been extracted from file, run the extract_abstract_from_pdf() class method first!')
        
        # Symbol replacements
        symbol_replacements = {
            '(cid:39)': '≃',
            '(cid:12)': 'Z⊙',
        }

        # Normalize Unicode characters to their base form
        self.text = unicodedata.normalize('NFKD', self.raw_text)

        # Replace symbol placeholders with their corresponding symbols
        for symbol, replacement in symbol_replacements.items():
            self.text = self.text.replace(symbol, replacement)

        # Remove excessive whitespaces
        self.text = re.sub(r'\s+', ' ', self.text)

        # Lowercase the text
        self.text = self.text.lower()

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(self.text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        self.text = ' '.join(lemmatized_words)
        
        # Replace astronomical abbreviations (list-limited)
        self.text = replace_astronomical_terms(self.text) if replace_astro_terms else self.text

        return 

    def summarize(self, max_length=512, min_length=30, do_sample=False):
        """
        Summarizes the given document based on the specified text section.
        This will update the ``df`` class attribute to contain only the author and summary.

        Args:
            max_length (int): The maximum length of the generated summary.
            min_length (int): The minimum length of the generated summary.
            do_sample (bool): Whether to use sampling during the summarization process.

        Returns:
            None
        """

        if len(self.filenames) == 0:
            raise ValueError('The filenames attribute is empty, run the save_paper() class method first!')
        
        summaries, authors, similarity_score = [], [], []

        progess_bar = bar.FillingSquaresBar('Summarizing.......', max=len(self.filenames))

        for fname in self.filenames:
            try:
                self.extract_abstract_from_pdf(filename=self.path+fname) #Creates the ``raw_text`` attribute
            except MissingFileError: 
                print(); print(f"WARNING: Could not find file: {self.path+fname}")

            if self.raw_text == '':
                print(f"Could not extract abstract for: {fname}")
                summaries.append('!!!Could not extract abstract!!!'); authors.append(fname)
                if self.user_input is not None:
                    similarity_score.append(self.is_related(generated_summary))
                continue
                
            self.process_abstract() #Creates the ``text`` attribute
         
            # Generate summary
            summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', revision='a4f8f3e')
        
            try:
                # Tokenize self.text
                tokenized_text = word_tokenize(self.text)

                # Truncate if necessary
                if len(tokenized_text) > max_length:
                    tokenized_text = tokenized_text[:max_length]

                # Convert tokenized text back to a string
                truncated_text = ' '.join(tokenized_text)

                # Generate summary and append
                summary = summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=do_sample)
                generated_summary = summary[0]['summary_text']
                summaries.append(generated_summary); authors.append(fname)

                if self.user_input is not None:
                    similarity_score.append(self.is_related(generated_summary))
            except Exception as e:
                print(f"Error occurred while summarizing {fname}: {str(e)}")

                if self.user_input is not None:
                    similarity_score.append(self.is_related(generated_summary))
            except Exception as e:
                print(f"Error occurred while summarizing {fname}: {str(e)}")

            progess_bar.next()

        progess_bar.finish()

        if self.user_input is None:
            self.df = pd.DataFrame({'Author': authors, 'Summary': summaries}, columns=['Author', 'Summary'])
        else:
            self.df = pd.DataFrame({'Author': authors, 'Summary': summaries, 'Similarity': similarity_score}, columns=['Author', 'Summary', 'Similarity'])
            self.df = self.df.sort_values(by='Similarity', ascending=False)

        self.df = self.df.reset_index(drop=True)

        return 

    def is_related(self, summary):
        """
        This function is_related calculates the cosine similarity between a user input and a summary text. 
        It uses the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to create TF-IDF matrices for the input and summary. 
        The cosine similarity is then calculated based on these matrices and returned.

        Args:
            summary: The summary text.

        Returns:
            The cosine similarity score.
        """

        if self.user_input is None:
            raise ValueError('The user_input attribute has not been input!')

        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Create TF-IDF matrices for the user input and summary text
        tfidf_matrix = vectorizer.fit_transform([self.user_input, summary])

        # Calculate the cosine similarity between the user input and summary
        similarity = cosine_similarity(tfidf_matrix)[0][1]

        return similarity

    def remove_irrelevant_papers(self, similarity_threshold=0):
        """
        Removes the papers that have similarity scores less than or equal to the specified similarity_threshold.

        Args:
            similarity_threshold
        """

        if self.df is None:
            raise ValueError('The df attribute does not yet exist, run the summarize() class method first!')

        indices = np.where(self.df.Similarity <= similarity_threshold)[0]

        for index in indices:
            try:
                os.remove(self.path+self.df.Author.iloc[index])
            except FileNotFoundError:
                pass 
                
        print(f"Complete! The following directory now contains only relevant papers: {self.path}")

        return 

def replace_astronomical_terms(text):
    """
    Replaces astronomical terms and abbreviations with their expanded forms or corresponding concepts.

    Args:
        text (str): The preprocessed text.

    Returns:
        str: The text with replaced astronomical terms.
    """

    replacements = {
        'eor': 'epoch of reionization',
        'fuv': 'far-ultraviolet',
        'uv': 'ultraviolet',
        'sfr': 'star formation rate',
        'sf': 'star-forming',
        'agn': 'active galactic nucleus',
        'sn': 'supernova',
        'laes': 'lyman alpha emitters',
        'cmb': 'cosmic microwave background',
        'iras': 'Infrared Astronomical Satellite',
        'sdss': 'Sloan Digital Sky Survey',
        'wfirst': 'Wide Field Infrared Survey Telescope',
        'lsst': 'Large Synoptic Survey Telescope',
        'alma': 'Atacama Large Millimeter Array',
        'vla': 'Very Large Array',
        'hst': 'Hubble Space Telescope',
        'jwst': 'James Webb Space Telescope',
        'wr': 'Wolf-Rayet',
    }

    for term, replacement in replacements.items():
        text = text.replace(term, replacement)
        text = text.replace(term.upper(), replacement)
        text = text.replace(term.capitalize(), replacement)

    return text