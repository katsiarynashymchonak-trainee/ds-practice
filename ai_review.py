import os
import openai
from github import Github
from github.Auth import Token

# Инициализация OpenAI и GitHub
openai.api_key = os.getenv("OPENAI_API_KEY")
github_token = os.getenv("MY_GITHUB_TOKEN")
g = Github(auth=Token(github_token))

# Получение репозитория
repo_name = os.getenv("GITHUB_REPOSITORY")
repo = g.get_repo(repo_name)

# Получение номера PR из переменной окружения
pr_number = os.getenv("PR_NUMBER")
if pr_number is None:
    print("❌ Переменная PR_NUMBER не установлена.")
    exit(1)

try:
    pr_number = int(pr_number)
except ValueError:
    print(f"❌ Невозможно преобразовать PR_NUMBER в число: {pr_number}")
    exit(1)

# Получение объекта PR
pr = repo.get_pull(pr_number)

# Получение diff-ссылки
diff = pr.diff_url

# Запрос к OpenAI
prompt = f"Проанализируй следующий diff и предложи улучшения:\n{diff}"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

# Публикация комментария в PR
comment = response.choices[0].message.content
pr.create_issue_comment(f"AI Review:\n{comment}")
