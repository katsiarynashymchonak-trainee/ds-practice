import os

from github import Github
from github.Auth import Token
from openai import OpenAI

# Инициализация OpenAI клиента
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Инициализация GitHub клиента
github_token = os.getenv("MY_GITHUB_TOKEN")
g = Github(auth=Token(github_token))

# Получение репозитория
repo_name = os.getenv("GITHUB_REPOSITORY")
repo = g.get_repo(repo_name)

# Получение номера PR из переменной окружения
pr_number = os.getenv("PR_NUMBER")
if pr_number is None:
    print("Переменная PR_NUMBER не установлена.")
    exit(1)

try:
    pr_number = int(pr_number)
except ValueError:
    print(f"Невозможно преобразовать PR_NUMBER в число: {pr_number}")
    exit(1)


# Получение объекта Pull Request
pr = repo.get_pull(pr_number)

# Получение ссылки на diff
diff_url = pr.diff_url

# Формирование запроса к OpenAI
prompt = f"Проанализируй следующий diff и предложи улучшения:\n{diff_url}"

response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": prompt}]
)

# Извлечение ответа и публикация комментария в PR
comment = response.choices[0].message.content
pr.create_issue_comment(f"AI Review:\n{comment}")
