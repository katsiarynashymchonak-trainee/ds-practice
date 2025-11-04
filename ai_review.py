import os
import requests
import google.generativeai as genai
from github import Github
from github.Auth import Token

# Gemini initialization
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-pro")

# GitHub initialisation
github_token = os.getenv("MY_GITHUB_TOKEN")
g = Github(auth=Token(github_token))

repo_name = os.getenv("GITHUB_REPOSITORY")
repo = g.get_repo(repo_name)

pr_number = os.getenv("PR_NUMBER")
if pr_number is None:
    print("PR_NUMBER environment variable is not set.")
    exit(1)

try:
    pr_number = int(pr_number)
except ValueError:
    print(f"Unable to convert PR_NUMBER to integer: {pr_number}")
    exit(1)

pr = repo.get_pull(pr_number)

# Get diff
diff_url = pr.diff_url
diff_response = requests.get(diff_url)
if diff_response.status_code != 200:
    print(f"Failed to fetch diff: {diff_response.status_code}")
    exit(1)

diff_content = diff_response.text

def split_diff(diff_text, max_lines=100):
    lines = diff_text.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i + max_lines])
        chunks.append(chunk)
    return chunks

reviews = []
chunks = split_diff(diff_content, max_lines=100)
for i, chunk in enumerate(chunks):
    prompt = f"Part {i+1} of {len(chunks)}. Analyze the following code diff and suggest improvements:\n{chunk}"
    try:
        response = model.generate_content(prompt)
        reviews.append(f"### Review Part {i+1}:\n{response.text}")
    except Exception as e:
        reviews.append(f"### Review Part {i+1}:\nError during analysis: {str(e)}")

full_review = "\n\n".join(reviews)
print("===== AI Review Output =====")
print(full_review)
print("============================")
pr.create_issue_comment(f"**AI Review Summary**\n\n{full_review}")
