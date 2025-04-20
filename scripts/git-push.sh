#!/bin/bash

# Get the commit message from the user
echo "Enter Commit Message:"
read commit_message

git add .
git commit -m "$commit_message"
git push origin main
