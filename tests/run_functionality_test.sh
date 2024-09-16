#!/bin/bash

max_retries=5
count=0
success=0
expected_output="Based on the articles, the presidential nominees are:

1. Republican nominee: Donald Trump
2. Democratic nominee: Kamala Harris"

while [ $count -lt $max_retries ]; do echo "Running functionality test (attempt $((count+1)))..." output=$(echo -e "spawn python3 ragnews.py; expect "ragnews> "; send "Who are the presidential nominees?\r"; expect "$expected_output"; interact" | expect)

if [[ $output == "$expected_output" ]]; then success=1 break fi

count=$((count+1)) echo "Functionality test failed. Retrying..." done

if [ $success -eq 0 ]; then echo "Functionality test failed after $max_retries attempts." exit 1 fi

echo "Functionality test passed."
