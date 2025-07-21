import subprocess
import sys
import os
import re
import requests
import random

from markdownify import markdownify as md

# --- [新增] 全局配置常量 ---
# 输入文件，脚本将检查此文件的 git diff
INPUT_FILE = "README.md"
# 自动归档的目标文件
CATEGORY_FILE = "category.md"


# --- [新增] 从环境变量获取 LLM API 配置 ---
def get_api_config() -> dict | None:
    """
    从环境变量中获取并验证 LLM API 配置。

    Returns:
        dict: 包含 api_url, api_key, model 的字典，如果缺少任何必要配置则返回 None。
    """
    # 从环境变量获取 API URL，这是必需的
    api_url = os.getenv("LLM_API_URL")
    if not api_url:
        print("错误: 缺少环境变量 'LLM_API_URL'。请设置 API 的请求地址。", file=sys.stderr)
        return None

    # 从环境变量获取 API Key，这是必需的
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误: 缺少环境变量 'OPENAI_API_KEY'。请设置您的 API 密钥。", file=sys.stderr)
        return None

    # 从环境变量获取模型名称，如果未设置则使用默认值
    model = os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

    return {"api_url": api_url, "api_key": api_key, "model": model}


# --- [新增] 从环境变量获取 Cloudflare API 配置 ---
def get_cloudflare_config() -> dict | None:
    """
    从环境变量中获取并验证 Cloudflare API 配置。

    Returns:
        dict: 包含 account_id 和 api_token 的字典，如果缺少则返回 None。
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        print("错误: 缺少环境变量 'CLOUDFLARE_ACCOUNT_ID'。无法使用 Cloudflare 获取微信文章。", file=sys.stderr)
        return None

    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not api_token:
        print("错误: 缺少环境变量 'CLOUDFLARE_API_TOKEN'。无法使用 Cloudflare 获取微信文章。", file=sys.stderr)
        return None

    return {"account_id": account_id, "api_token": api_token}


# --- [已修改] 使用环境变量配置的 OpenAI API 函数 ---
def summarize_with_openai(content: str) -> str | None:
    """
    使用配置好的 OpenAI API 为给定文本生成摘要。
    """
    config = get_api_config()
    if not config:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    system_prompt = "你是一位专业的文章摘要助手。请将以下文章内容生成一段精炼的中文摘要，要求语言流畅、抓住核心要点，并严格控制在150个字以内。"
    payload = {
        "model": config['model'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0.3,
    }

    try:
        print("    > 正在通过 requests 请求 LLM 生成摘要...")
        response = requests.post(config['api_url'], headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        response_data = response.json()
        summary = response_data['choices'][0]['message']['content']
        return summary.strip()
    except requests.RequestException as e:
        print(f"    > LLM API 请求失败 (网络错误): {e}", file=sys.stderr)
        if e.response is not None:
             print(f"    > 响应内容: {e.response.text}", file=sys.stderr)
        return None
    except (KeyError, IndexError) as e:
        print(f"    > 解析 LLM 响应失败: 意外的格式。错误: {e}", file=sys.stderr)
        return None


# --- [已修改] 使用环境变量配置的 AI 分类函数 ---
def categorize_with_openai(title: str, summary: str, existing_categories: list[str]) -> str | None:
    """
    使用配置好的 OpenAI API 对文章进行分类。
    """
    config = get_api_config()
    if not config:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    category_list_str = "\n".join(f"- {cat}" for cat in existing_categories)
    system_prompt = (
        "你是一位智能分类助手。你的任务是根据文章的标题和摘要，将其分配到一个最合适的类别中。"
        "请严格从以下【已有类别】列表中选择一个。如果所有类别都不太合适，请创造一个新的、简洁的类别名称（例如 '云原生技术' 或 '产品与设计'）。"
        "你的回答必须且只能是类别名称本身，不要包含任何多余的文字、解释或标点符号（如 '类别：' 或 '##'）。"
    )
    user_content = f"""
【已有类别】:
{category_list_str}

【文章标题】:
{title}

【文章摘要】:
{summary}

请为以上内容选择或创建一个最合适的类别名称：
"""
    payload = {
        "model": config['model'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()}
        ],
        "temperature": 0.1,
    }

    try:
        print("    > 正在通过 requests 请求 LLM 进行分类...")
        response = requests.post(config['api_url'], headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        response_data = response.json()
        category = response_data['choices'][0]['message']['content'].strip()
        # 移除AI可能返回的多余字符
        category = re.sub(r'^[#*"\s]+|[#*"\s]+$', '', category)
        return category
    except requests.RequestException as e:
        print(f"    > LLM API 分类请求失败 (网络错误): {e}", file=sys.stderr)
        if e.response is not None:
             print(f"    > 响应内容: {e.response.text}", file=sys.stderr)
        return None
    except (KeyError, IndexError) as e:
        print(f"    > 解析 LLM 分类响应失败: 意外的格式。错误: {e}", file=sys.stderr)
        return None


# --- [新增] 使用 Cloudflare 获取微信公众号内容 ---
def fetch_content_with_cloudflare(url: str) -> str | None:
    """
    使用 Cloudflare 浏览器渲染和 AI Markdown 转换获取文章内容。
    专为解决微信公众号等难以抓取的网站设计。
    """
    config = get_cloudflare_config()
    if not config:
        return None

    account_id = config["account_id"]
    api_token = config["api_token"]

    headers = {"Authorization": f"Bearer {api_token}"}

    # 第 1 步: 使用浏览器渲染获取 HTML
    render_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/browser-rendering/content"
    render_payload = {"url": url}
    
    try:
        print(f"    > 正在通过 Cloudflare 浏览器渲染获取 HTML: {url}")
        # 渲染可能耗时较长，设置更长的超时时间
        response = requests.post(render_url, headers={"Content-Type": "application/json", **headers}, json=render_payload, timeout=600)
        response.raise_for_status()
        render_data = response.json()

        if not render_data.get("success"):
            print(f"    > Cloudflare 浏览器渲染失败: {render_data.get('errors')}", file=sys.stderr)
            return None
        
        html_content = render_data['result']

    except requests.RequestException as e:
        print(f"    > Cloudflare 浏览器渲染请求失败: {e}", file=sys.stderr)
        if e.response is not None:
             print(f"    > 响应内容: {e.response.text}", file=sys.stderr)
        return None
    except (KeyError, TypeError):
        print(f"    > Cloudflare 浏览器渲染响应格式不正确。", file=sys.stderr)
        return None

    # 第 2 步: 将 HTML 转换为 Markdown
    return md(html_content).strip()


# --- Jina Reader 函数 (保持不变) ---
def fetch_content_with_jina(url: str) -> str | None:
    jina_reader_url = f"https://r.jina.ai/{url}"
    headers = {"Accept": "text/plain", "User-Agent": "MyBookmarkProcessor/1.0"}
    try:
        print(f"    > 正在通过 Jina Reader 获取内容: {url}")
        response = requests.get(jina_reader_url, headers=headers, timeout=60)
        response.raise_for_status()
        full_text = response.text
        if "Markdown Content:\n" in full_text:
            content_part = full_text.split("Markdown Content:\n", 1)[1]
            return content_part.strip()
        else:
            print("    > 警告: Jina Reader 未返回预期的 'Markdown Content:' 格式。", file=sys.stderr)
            return full_text.strip()
    except requests.RequestException as e:
        print(f"    > Jina Reader API 请求失败: {e}", file=sys.stderr)
        return None


# --- [新增] 内容获取调度函数 ---
def fetch_article_content(url: str) -> str | None:
    """
    根据 URL 类型选择合适的抓取器 (Cloudflare 或 Jina)。
    优先处理微信公众号链接。
    """
    if "mp.weixin.qq.com" in url:
        print("  > 检测到微信公众号链接，将使用 Cloudflare 抓取...")
        return fetch_content_with_cloudflare(url)
    else:
        print("  > 使用 Jina Reader 抓取...")
        return fetch_content_with_jina(url)


# --- Git 和解析相关的函数 (保持不变) ---
def parse_markdown_links_from_diff(diff_text: str) -> list:
    link_pattern = re.compile(r'\[(.+?)\]\((.+?)\)')
    extracted_links = []
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            content = line[1:].strip()
            matches = link_pattern.findall(content)
            for title, url in matches:
                extracted_links.append({'title': title.strip(), 'url': url.strip()})
    return extracted_links

def get_file_last_change_diff_text(file_path: str, repo_path: str) -> str | None:
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        print(f"错误: '{repo_path}' 不是一个有效的 Git 仓库。", file=sys.stderr)
        return None
    try:
        log_command = ['git', 'log', '-n', '2', '--pretty=%H', '--', file_path]
        result = subprocess.run(log_command, cwd=repo_path, capture_output=True, text=True, check=True)
        hashes = [h for h in result.stdout.strip().split('\n') if h]
    except subprocess.CalledProcessError as e:
        print(f"错误：无法获取 '{file_path}' 的提交历史: {e.stderr.strip()}", file=sys.stderr)
        return None
    if len(hashes) < 2:
        print(f"文件 '{file_path}' 只有一个或没有变更历史，无法进行对比。")
        return None
    newer_commit, older_commit = hashes[0], hashes[1]
    print(f"对比文件 '{file_path}' 的最后两次变更 (from {older_commit[:7]} to {newer_commit[:7]})")
    print("=" * 60)
    try:
        diff_command = ['git', 'diff', older_commit, newer_commit, '--', file_path]
        diff_result = subprocess.run(diff_command, cwd=repo_path, capture_output=True, text=True, check=True)
        return diff_result.stdout
    except subprocess.CalledProcessError as e:
        print(f"错误：执行 git diff 失败: {e.stderr.strip()}", file=sys.stderr)
        return None


# --- 文件处理函数 ---
def parse_categories_from_file(file_path: str) -> list[str]:
    """
    从文件中解析H2(##)和H3(###)标题作为分类。
    """
    if not os.path.exists(file_path):
        return []
    categories = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('## ') or line.startswith('### '):
                    header_content = line.lstrip('#').strip()
                    parts = header_content.split(maxsplit=1)
                    category_name = parts[-1]
                    categories.append(category_name)
    except IOError as e:
        print(f"错误: 无法读取分类文件 '{file_path}': {e}", file=sys.stderr)
    return categories

def insert_article_to_category_file(file_path: str, category: str, title: str, url: str, summary: str):
    """
    将文章插入到指定的分类下，支持H2和H3级别的分类。
    """
    try:
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# 网站资源分类整理\n\n")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"错误: 读写文件 '{file_path}' 失败: {e}", file=sys.stderr)
        return

    article_text = f"**标题:** {title}\n\n**链接:** {url}\n\n**摘要:** {summary}"
    category_header_index = -1

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('## ') or stripped_line.startswith('### '):
            header_content = stripped_line.lstrip('#').strip()
            parts = header_content.split(maxsplit=1)
            name_in_header = parts[-1]
            if name_in_header == category:
                category_header_index = i
                break

    if category_header_index != -1:
        print(f"    > 分类 '{category}' 已存在，正在查找插入位置...")
        first_article_index = -1
        for i in range(category_header_index + 1, len(lines)):
            if lines[i].strip().startswith("**标题:**"):
                first_article_index = i
                break

        if first_article_index != -1:
            insertion_text = f"{article_text}\n\n---\n\n"
            lines.insert(first_article_index, insertion_text)
            print(f"    -> 成功将文章插入到 '{category}' 分类顶部。")
        else:
            end_of_section_index = len(lines)
            for i in range(category_header_index + 1, len(lines)):
                if lines[i].startswith("##"):
                    end_of_section_index = i
                    break
            insertion_text = f"\n{article_text}\n"
            lines.insert(end_of_section_index, insertion_text)
            print(f"    -> 成功将文章添加到空的 '{category}' 分类下。")
    else:
        print(f"    > 分类 '{category}' 是新分类，将在文件末尾创建。")
        if lines and not lines[-1].endswith('\n'):
            lines.append("\n")
        if lines and lines[-1].strip() != "":
            lines.append("\n")
        emojis = ["🧩", "🔧", "💡", "📚", "🧭", "✨"]
        new_category_header = f"## {random.choice(emojis)} {category}\n"
        lines.append(new_category_header)
        lines.append("\n")
        lines.append(f"{article_text}\n")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except IOError as e:
        print(f"错误: 写入文件 '{file_path}' 失败: {e}", file=sys.stderr)


# --- [已修改] 主程序入口 ---
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动自动化书签处理与归档脚本 🚀")
    
    # 检查 API 配置是否齐全
    api_config = get_api_config()
    if not api_config:
        print("启动失败：请先根据提示设置好必要的 LLM 环境变量。", file=sys.stderr)
        sys.exit(1)

    print(f"  - 输入文件: {INPUT_FILE}")
    print(f"  - 归档文件: {CATEGORY_FILE}")
    print(f"  - LLM 模型: {api_config['model']}")
    print("-" * 60)

    # 仓库路径，默认为当前目录，也可通过环境变量配置
    repository_path = os.getenv("GIT_REPO_PATH", ".")
    
    # 获取文件变更
    diff_output = get_file_last_change_diff_text(INPUT_FILE, repository_path)

    if diff_output:
        extracted_links = parse_markdown_links_from_diff(diff_output)

        if not extracted_links:
            print("在本次变更中没有找到新增的 Markdown 链接。")
        else:
            print(f"解析到 {len(extracted_links)} 个新增链接，正在处理...\n")
            
            print(f"--- 准备工作 ---")
            print(f"  > 正在从 '{CATEGORY_FILE}' 解析现有分类...")
            existing_categories = parse_categories_from_file(CATEGORY_FILE)
            if existing_categories:
                print(f"  > 已找到 {len(existing_categories)} 个分类: {', '.join(existing_categories)}\n")
            else:
                print(f"  > 未找到任何现有分类，将由 AI 自动创建。\n")

            for i, link_data in enumerate(extracted_links):
                print(f"--- 处理第 {i+1}/{len(extracted_links)} 个链接 ---")
                print(f"  原始标题: {link_data['title']}")
                print(f"  原始链接: {link_data['url']}")

                # [修改] 使用新的调度函数获取内容
                content = fetch_article_content(link_data['url'])

                if content:
                    summary = summarize_with_openai(content)
                    if summary:
                        print(f"  AI 摘要: {summary}")
                        
                        chosen_category = categorize_with_openai(
                            link_data['title'], 
                            summary, 
                            existing_categories
                        )
                        
                        if chosen_category:
                            print(f"  AI 分类: {chosen_category}")
                            insert_article_to_category_file(
                                CATEGORY_FILE,
                                chosen_category,
                                link_data['title'],
                                link_data['url'],
                                summary
                            )
                            # 如果AI创建了一个全新的分类，将其加入列表，供后续链接使用
                            if chosen_category not in existing_categories:
                                existing_categories.append(chosen_category)
                            print(f"  [成功] 文章已自动归档到 '{CATEGORY_FILE}'。\n")
                        else:
                            print("  [失败] AI 分类生成失败，跳过归档。\n")
                    else:
                        print("  AI 摘要: 生成失败。\n")
                else:
                    print("  内容获取: 失败，跳过摘要生成。\n")
            print("=" * 60)
            print("所有链接处理完毕。")
    print("=" * 60)