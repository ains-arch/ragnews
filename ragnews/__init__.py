#!/bin/python3
# pylint: disable=C0301

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented
generation (RAG).

The path to the database can be changed with the --db parameter, but it defaults to
databases/ragnews.db
'''

import logging
import re
import sqlite3
import os
import time
import groq


################################################################################
# LLM functions
################################################################################

client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def split_document_into_chunks(text, max_chunk_size=4000):
    r"""
    Split the input text into a list of smaller chunks of text so that an LLM
    can process the chunks individually.

    Arguments:
        text (str): The original document to split up.
        max_chunk_size (int): The number of words in a chunk.
    
    Returns:
        chunks (list): The document as a series of strings in a list.

    >>> split_document_into_chunks('This is a paragraph.')
    ['This is a paragraph.']

    >>> split_document_into_chunks('This is a paragraph.\n\nThis is another paragraph.')
    ['This is a paragraph.\n\nThis is another paragraph.']

    >>> split_document_into_chunks('This is a paragraph.\n\nThis is another paragraph.\n\nThis is yet another paragraph.')
    ['This is a paragraph.\n\nThis is another paragraph.\n\nThis is yet another paragraph.']

    >>> doc = split_document_into_chunks('This is a paragraph.\n\nThis is another paragraph.\n\nThis is yet another paragraph.\n\nThis is too long to fit in one chunk.' * 1000, max_chunk_size=500)
    >>> len(doc[0])
    496
    >>> len(doc)
    250

    >>> split_document_into_chunks('')
    []

    >>> split_document_into_chunks('This is a paragraph.\n\n', max_chunk_size=10)
    ['This is a paragraph.']

    >>> split_document_into_chunks('Short text', max_chunk_size=10)
    ['Short text']
    """
    # Return empty list if the input text is empty
    if not text:
        return []

    # Split the document by two or more newlines
    paragraphs = re.split(r'\n{2,}', text)
    # print(f"DEBUG: paragraphs: {paragraphs}")

    # Remove leading/trailing newlines and spaces from each paragraph
    cleaned_paragraphs = [para.strip() for para in paragraphs if para.strip()]
    # print(f"DEBUG: cleaned_paragraphs: {cleaned_paragraphs}")

    chunks = []
    # print(f"DEBUG: chunks: {chunks}")
    current_chunk = ""
    # print(f"DEBUG: current_chunk: {current_chunk}")

    for para in cleaned_paragraphs:
        # print(f"DEBUG: para: {para}")
        if len(current_chunk) + len(para) + 2 > max_chunk_size:
            # If too long, add the current chunk to the list and move on
            # +2 for potential newlines
            # print("DEBUG: too long")
            if current_chunk:
                # Don't add an empty string
                chunks.append(current_chunk.strip())
            # print(f"DEBUG: chunks: {chunks}")
            current_chunk = para
        else:
            # print("DEBUG: keep adding")
            # Otherwise, keep adding to the current chunk
            if current_chunk:
                current_chunk += "\n\n"  # Add a separator between paragraphs
                # print(f"DEBUG: current_chunk: {current_chunk}")
            current_chunk += para
            # print(f"DEBUG: current_chunk: {current_chunk}")

    # If anything got added to the current chunk string
    if current_chunk:
        # print(f"DEBUG: current_chunk exists: {current_chunk}")
        # Add it to the chunks list
        chunks.append(current_chunk.strip())
        # print(f"DEBUG: chunks: {chunks}")

    # Return the document as the series of chunks
    return chunks

def run_llm(system, user, model='llama-3.1-70b-versatile', seed=None, delay=5):
    """
    This is a helper function for all the uses of LLMs in this file.
    
    Arguments:
        system (str): The system prompt to pass to the model.
        user (str): The user prompt to pass to the model.
        model (str): The name of the model to use from the Groq API.
        seed (int): An integer used to constrain Groq's randomness into near-determinism.
    
    Returns:
        chat_completion.choices[0].message.content (str): Response string from Groq API.
    """
    try:
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
    except groq.InternalServerError:
        # Situations where the probably is just the Groq API being unstable
        # print("DEBUG: Groq broke, try again")
        # The solution is just to wait and try again
        time.sleep(delay)
        chat_completion = run_llm(system, user)

    except (groq.RateLimitError, groq.BadRequestError) as e:
        # Situations where either
        #   The individual query itself is too long (BadRequestError)
        #   The individual query itself (and possibly the previous queries
        #       from the last minute) are cumulatively too long
        #       (RateLimitError, 'Tokens Per Minute (TMP)')
        #   There have been too many queries in the past minute
        #       (RateLimitError, 'Requests Per Minute (RMP)')
        # In the first two situations, the solution is to break up the current
        #   query into smaller chunks and recursively query the API again.
        # In the last situation, the solution is to wait and query the API
        #   again.

        error_message = str(e)

        if 'RMP' in error_message:
            # Situations where the there have been too many queries in the
            # past minute
            # print(f"DEBUG: Too many queries, waiting {delay} seconds")
            time.sleep(delay)
            chat_completion = run_llm(system, user)

        else:
            # Split the document into chunks
            chunked_text = split_document_into_chunks(user)
            # print(f"DEBUG: number of chunks: {len(chunked_text)}")

            # Initialize an empty list for storing individual summaries
            summarized_chunks = []

            # Summarize each paragraph
            for chunk in chunked_text:
                # print(f"DEBUG: chunk {i}")
                # print(f"DEBUG: length of chunk: {len(chunk)}")
                # print(f"DEBUG: chunk: {chunk}")

                # Get the query object (this function only returns the object)
                chunk_chat = run_llm(system, chunk)

                # Get the message text from the query object
                chunk_txt = chunk_chat.choices[0].message.content

                # print(f"DEBUG: internal response: {chunk_txt}")

                # Add it to the summaries list
                summarized_chunks.append(chunk_txt)

            # Concatenate all summarized paragraphs into a smaller document
            summarized_document = " ".join(summarized_chunks)

            # print(f"DEBUG: len big summary: {len(summarized_document)}")
            # print(f"DEBUG: summarized_document: {summarized_document}")
            # Submit the concatenated summary for summary itself

            chat_completion = run_llm(system, summarized_document)

    # Return the API chat object for the summary of the full text
    return chat_completion

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
    return run_llm(system, text, seed=seed).choices[0].message.content

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
    return run_llm(system, text).choices[0].message.content


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

    return run_llm(system, text, seed=seed).choices[0].message.content


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


def rag(text, db, keywords_text=None):
    '''
    This function uses retrieval augmented generation (RAG) to generate an LLM response to the input
    text. The db argument should be an instance of the `ArticleDB` class that contains the relevant
    documents to use.

    NOTE:
    There are no test cases because:
    1. the answers are non-deterministic (both because of the LLM and the database), and
    2. evaluating the quality of answers automatically is non-trivial.
    '''

    if not keywords_text:
        keywords_text=text
    # Extract keywords from the input text (user's question)
    bad_keywords = extract_keywords(keywords_text)
    # print(f"DEBUG: bad keywords: {bad_keywords}")
    keywords = re.sub(r'[^a-zA-Z0-9\s]', '', bad_keywords)
    # print(f"DEBUG: keywords: {keywords}")

    # print(f"length of database: {len(db)}")


    # Search for relevant articles in the database using the keywords
    articles = db.find_articles(keywords)
    # print(f"DEBUG: articles: {articles}")

    # Construct the new prompt with articles and user's question
    articles_content = "\n".join([
        f"ARTICLE{index}_URL: {article['url']}\n"
        f"ARTICLE{index}_TITLE: {article['title']}\n"
        f"ARTICLE{index}_SUMMARY: {article['en_summary']}"
        for index, article in enumerate(articles)
    ])
    # print(f"DEBUG: articles_content: {articles_content}")


    system_prompt = '''You are a news analyst. You will be given several articles and question
    Answer the question based on the articles.'''

    new_prompt = f'''
    
    {articles_content}

    QUESTION: {text}
    '''
    # print(f"DEBUG: new_prompt: {new_prompt}")

    # Pass the new prompt to the LLM
    response = run_llm(system_prompt, new_prompt).choices[0].message.content

    return response

class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    We can search through those articles to find articles about only certain topics.

    >>> articles = db.find_articles('For example, a timely question about American politics')

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
    parser.add_argument('--db', default='ragnews.db')
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
