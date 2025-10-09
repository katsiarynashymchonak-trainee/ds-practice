import os
import openai
from github import Github

openai.api_key = os.getenv("OPENAI_API_KEY")
g = Github(os.getenv("MY_GITHUB_TOKEN"))

repo_name = os.getenv("GITHUB_REPOSITORY")
repo = g.get_repo(repo_name)
pr_number = int(os.getenv("GITHUB_REF").split("/")[-1])
pr = repo.get_pull(pr_number)

diff = pr.diff_url  # или получи изменения через pr.get_files()

prompt = f"Проанализируй следующий diff и предложи улучшения:\n{diff}"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

comment = response.choices[0].message.content
pr.create_issue_comment(f"AI Review:\n{comment}")
