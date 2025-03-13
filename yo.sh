#!/bin/bash
set -e

# Define the old email and the new details.
OLD_EMAIL="jonasGruttergmail.com"
NEW_EMAIL="jonasgrutter@gmail.com"
NEW_NAME="JonasGrutter"

# Rewrite the commit history
git filter-branch --env-filter '
if [ "$GIT_AUTHOR_EMAIL" = "'"$OLD_EMAIL"'" ]; then
    export GIT_AUTHOR_NAME="'"$NEW_NAME"'"
    export GIT_AUTHOR_EMAIL="'"$NEW_EMAIL"'"
fi
if [ "$GIT_COMMITTER_EMAIL" = "'"$OLD_EMAIL"'" ]; then
    export GIT_COMMITTER_NAME="'"$NEW_NAME"'"
    export GIT_COMMITTER_EMAIL="'"$NEW_EMAIL"'"
fi
' --tag-name-filter cat -- --branches --tags

echo "Author and committer information updated successfully."
