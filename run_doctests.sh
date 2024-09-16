#!/bin/bash

max_retries=5
count=0
success=0

while [ $count -lt $max_retries ]; do
  echo "Running doctests (attempt $((count+1)))..."
  python3 -m doctest ragnews.py && { success=1; break; }
  count=$((count+1))
  echo "Doctests failed. Retrying..."
done

if [ $success -eq 0 ]; then
  echo "Doctests failed after $max_retries attempts."
  exit 1
fi

echo "Doctests passed."
