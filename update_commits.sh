#!/bin/bash
# This script updates commit author email for commits by JonasGrutter with the wrong email.

# Step 1: Extract all commit hashes where the author is "JonasGrutter <jonasgruttergmail.com>" (missing @)
git log --author='JonasGrutter <jonasgruttergmail.com>' --format='%H' > commit_hashes.txt

echo "Found the following commit hashes:"
cat commit_hashes.txt

# Step 2: Loop through each commit hash and update the commit author using git blame-someone-else
while IFS= read -r commit; do
    if [ -n "$commit" ]; then
        echo "Processing commit: $commit"
        git blame-someone-else "JonasGrutter <jonasGrutter@gmail.com>" "$commit"
    fi
done < commit_hashes.txt

echo "Done updating commits."
