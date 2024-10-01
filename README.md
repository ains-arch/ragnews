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

### Requirements

- Python 3.9

### Setup Instructions

1. **Add the deadsnakes PPA and install Python 3.9:**

```
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
$ sudo apt install python3.9 python3.9-venv python3.9-dev
```

2. **Create and activate a virtual environment:**

```
$ python3.9 -m venv venv
$ . ./venv/bin/activate
```

3. **Install the required Python packages:**

```
$ pip3 install -r requirements.txt
```

4. **Configure environment variables:**

    - Edit the `.env` file to include your Groq API key.
    - Export the variables:

        ```
        $ export $(cat .env)
        $ export PYTHONPATH=.
        ```

### Example Usage

```
$ python3 ragnews/__init__.py --db=ragnews/ragnews.db 
ragnews> Who are the major party presidential nominees?
According to the article, the major party presidential nominees are Donald Trump and Kamala Harris.
```

```
$ python3 ragnews/evaluate.py --path hairy-trumpet/data/'wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]'
0.8188976377952756
```
