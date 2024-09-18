#!/bin/python3
# pylint: disable=C0301

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented
generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

import logging
import re
import sqlite3
import os
from groq import Groq


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def run_llm(system, user, model='llama3-8b-8192', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                "role": "user",
                "content": user,
            }
        ],
        model=model,
        seed=seed, # specify seed of random number generator it's using
    )
    return chat_completion.choices[0].message.content


def summarize_text(text, seed=None):
    """
    Wrapper for the run_llm() function that includes a system prompt to tell Groq to summarize the
    input text.
    
    Arguments:
        text (str): The text to summarize.
        seed (int): An integer used to constrain Groq's randomness into near-determinism.
    
    Returns:
        run_llm() (function call): Queries Groq to return a Groq ChatCompletion object.
    """
    system = '''
    Summarize the input text below.  Limit the summary to 1 paragraph.
    Use an advanced reading level similar to the input text, and ensure that all people, places,
    and other proper and dates nouns are included in the summary.  The summary should be in English.
    '''
    return run_llm(system, text, seed=seed)


def translate_text(text):
    """
    Wrapper for the run_llm() function that includes a system prompt to tell Groq to translate the
    input text.
    
    Arguments:
        text (str): The text to translate.
    
    Returns:
        run_llm() (function call): Queries Groq to return a Groq ChatCompletion object.
    """
    system = '''You are a professional translator working for the United Nations.  The following
    document is an important news article that needs to be translated into English.  Provide a
    professional translation.'''
    return run_llm(system, text)


def extract_keywords(text, seed=None):
    r'''
    This is a helper function for RAG. Given an input text, this function extracts the keywords that
    will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Who is the current democratic presidential nominee?', seed=0)
    'democratic nomination democratic primary election democratic party candidate joe biden'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'Trump illegal Mexican immigration policy stance border control deportation'

    Note that the examples above are passing in a seed value for deterministic results.
    In production, you probably do not want to specify the seed.
    '''

    system = '''Respond with exactly ten search keywords from and related to the input
    below. Do not attempt to answer questions using the keywords. Stay focused
    on providing keywords from the question and keywords that describe the
    general topic. Do not include new lines or bullet points. Format your
    response like 'word word word word', with exactly 10 words.
    '''

    return run_llm(system, text, seed=seed)


################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\W+')
    sql_dewhite = rex.sub(' ', sql)
    logging.debug("SQL: %s", sql_dewhite)


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    We use this decorator on the add_urls method below to ensure that a webcrawl continues even if
    there are errors.
    '''
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e: # pylint: disable=W0718
            logging.error(str(e))
    return inner_function


################################################################################
# rag
################################################################################


def rag(text, db):
    '''
    This function uses retrieval augmented generation (RAG) to generate an LLM response to the input
    text. The db argument should be an instance of the `ArticleDB` class that contains the relevant
    documents to use.

    NOTE:
    There are no test cases because:
    1. the answers are non-deterministic (both because of the LLM and the database), and
    2. evaluating the quality of answers automatically is non-trivial.

    '''

    # Extract keywords from the input text (user's question)
    keywords = extract_keywords(text)

    # Search for relevant articles in the database using the keywords
    articles = db.find_articles(keywords)

    # Construct the new prompt with articles and user's question
    articles_content = "\n".join([
        f"ARTICLE{index}_URL: {article['url']}\n"
        f"ARTICLE{index}_TITLE: {article['title']}\n"
        f"ARTICLE{index}_SUMMARY: {article['en_summary']}"
        for index, article in enumerate(articles)
    ])


    system_prompt = '''You are a news analyst. You will be given several articles and question
    Answer the question based on the articles.'''

    new_prompt = f'''
    
    {articles_content}

    QUESTION: {text}
    '''

    # Pass the new prompt to the LLM

    response = run_llm(system_prompt, new_prompt)

    return response

class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    We can search through those articles to find articles about only certain topics.

    >>> articles = db.find_articles('Economía')

    The output is a list of articles that match the search query.
    Each article is represented by a dictionary with a number of fields about the article.
    '''

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory=sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.

        The test below demonstrates that creating a schema on a database that already has the schema
        will not generate errors.

        >>> db = ArticleDB()
        >>> db._create_schema()
        >>> db._create_schema()
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE articles
            USING FTS5 (
                title,
                text,
                hostname,
                url,
                publish_date,
                crawl_date,
                lang,
                en_translation,
                en_summary
                );
            '''
            self.db.execute(sql)
            self.db.commit()

        # if the database already exists,
        # then do nothing
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed')

    def find_articles(self, query, limit=8, timebias_alpha=1): # pylint: disable=W0613
        '''
        Return a list of articles in the database that match the specified query.

        Lowering the value of the timebias_alpha parameter will result in the time becoming more
        influential.

        The final ranking is computed by the FTS5 rank * timebias_alpha / (days since article
        publication + timebias_alpha).
        '''
        # TODO: what does the timebias_alpha param actually do here, it
        # doesn't appear to be being passed to anything

        sql = f'''
        SELECT url, title, en_summary
        FROM articles
        WHERE articles MATCH '{query}'
        ORDER BY rank
        LIMIT {limit};
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        # Get column names from cursor description
        columns = [column[0] for column in cursor.description]
        # Convert rows to a list of dictionaries
        row_dict = [dict(zip(columns, row)) for row in rows]
        return row_dict

    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM articles
        WHERE text IS NOT NULL;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='databases/ragnews.db')
    main_args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=main_args.loglevel.upper(),
        )

    main_db = ArticleDB(main_args.db)

    while True:
        user_text = input('ragnews> ')
        if len(user_text.strip()) > 0:
            output = rag(user_text, main_db)
            print(output)
