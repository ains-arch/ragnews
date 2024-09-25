# RAG News
![](https://github.com/ains-arch/ragnews/workflows/tests/badge.svg)

## Overview

`ragnews` is a Python-based question and answer system built on the Groq
API. It extends the functionality of the Groq models by augmenting user
prompts using RAG. The application fetches and processes news articles
from a user-provided database, then uses them to answer user queries
with a particular focus on providing accurate answers to timely
questions.

## Getting Started

To get started with `ragnews`, follow these steps to set up your
development environment and run the application:

### Setup Instructions

1. **Create and activate a virtual environment:**

```
$ python3 -m venv venv3.12
$ . ./venv3.12/bin/activate
```

1. **Install the required Python packages:**

```
$ pip3 install -r requirements.txt
```

1. **Configure environment variables:**

    - Edit the `.env` file to include your Groq API key and OpenAI API key.
    - Export the variables:

        ```
        $ export $(cat .env)
        ```

### Example Usage

```
$ python3 ragnews.py 
ragnews> Who are the presidential nominees?
Based on the article, the presidential nominees are Donald Trump and Kamala Harris
```
