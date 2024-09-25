from openai import OpenAI
client = OpenAI()

'''
response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        messages = [
            {
                'role': 'system',
                'content': 'You are an expert on politics. Answer the user question in one sentence.'
            },
            {
                'role': 'user',
                'content': 'Who are the current presidential candidates?'
            }
        ],
        )
'''

import lmql

# quotation marks are passed to llm
# [] are returned from llm
query = r'''

sample(temperature=0.8)
    "You are an expert on politics. Answer the user question in one sentence.\n"
    "Q: What are the main issues in the current election?\n"
    "A: The main issues are in the following list:\n"
    ISSUES = []
    for i in range(10):
        "-[ISSUE]" where STOPS_AT(ISSUE, "\n")
        ISSUES.append(ISSUE)
    return ISSUES
from
    "openai/gpt-3.5-turbo-instruct"

'''

result = lmql.run_sync(query)
