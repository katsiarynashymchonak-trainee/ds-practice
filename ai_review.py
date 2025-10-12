import os
import requests

from github import Github
from github.Auth import Token
from openai import OpenAI

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize GitHub client
github_token = os.getenv("MY_GITHUB_TOKEN")
g = Github(auth=Token(github_token))

# Get repository name from environment
repo_name = os.getenv("GITHUB_REPOSITORY")
repo = g.get_repo(repo_name)

# Get pull request number from environment
pr_number = os.getenv("PR_NUMBER")
if pr_number is None:
    print("PR_NUMBER environment variable is not set.")
    exit(1)

try:
    pr_number = int(pr_number)
except ValueError:
    print(f"Unable to convert PR_NUMBER to integer: {pr_number}")
    exit(1)

# Retrieve the pull request object
pr = repo.get_pull(pr_number)

# Download the diff content
diff_url = pr.diff_url
diff_response = requests.get(diff_url)
if diff_response.status_code != 200:
    print(f"Failed to fetch diff: {diff_response.status_code}")
    exit(1)

diff_content = diff_response.text

# Split diff into manageable chunks
def split_diff(diff_text, max_lines=100):
    lines = diff_text.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i + max_lines])
        chunks.append(chunk)
    return chunks

# Send each chunk to OpenAI for review
reviews = []
chunks = split_diff(diff_content, max_lines=100)
for i, chunk in enumerate(chunks):
    prompt = f"Part {i+1} of {len(chunks)}. Analyze the following diff and suggest improvements:\n{chunk}"
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        reviews.append(f"### Review Part {i+1}:\n{response.choices[0].message.content}")
    except Exception as e:
        reviews.append(f"### Review Part {i+1}:\nError during analysis: {str(e)}")

# Post the final review as a comment on the pull request
full_review = "\n\n".join(reviews)
pr.create_issue_comment(f"**AI Review Summary**\n\n{full_review}")
