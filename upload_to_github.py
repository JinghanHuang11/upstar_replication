#!/usr/bin/env python3
"""
upload_to_github.py

将项目代码上传到 GitHub 的 main 分支。
用法：python upload_to_github.py
"""

import subprocess
import sys
import os

# ============================================================
# 配置区（按需修改）
# ============================================================
GITHUB_USERNAME = "JinghanHuang11"
REPO_NAME = "upstar_replication"
GITHUB_TOKEN = ""  # 留空则从环境变量 GITHUB_TOKEN 读取
BRANCH = "main"
# ============================================================


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """运行 shell 命令并打印输出。"""
    print(f">>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, encoding="utf-8"
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        print(f"[ERROR] 命令失败，退出码 {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def get_token() -> str:
    token = GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("[ERROR] 未配置 GitHub Token。")
        print("  方式 1：在脚本顶部 GITHUB_TOKEN 变量填入 token")
        print("  方式 2：设置环境变量 GITHUB_TOKEN=<your_token>")
        sys.exit(1)
    return token


def main():
    token = get_token()
    remote_url = (
        f"https://{GITHUB_USERNAME}:{token}"
        f"@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    )

    print("\n=== UPSTAR GitHub 上传脚本 ===")
    print(f"目标仓库: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    print(f"目标分支: {BRANCH}\n")

    # 1. 确保当前在项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"工作目录: {script_dir}\n")

    # 2. 初始化 git（如果尚未初始化）
    if not os.path.isdir(".git"):
        run("git init")
        run(f'git config user.name "{GITHUB_USERNAME}"')
        run(f'git config user.email "{GITHUB_USERNAME}@users.noreply.github.com"')
    else:
        print(">>> git 仓库已存在，跳过 init")

    # 3. 切换 / 创建 main 分支
    current = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True, capture_output=True, text=True
    ).stdout.strip()
    if current != BRANCH:
        result = run(f"git checkout -B {BRANCH}", check=False)
        if result.returncode != 0:
            run(f"git checkout -b {BRANCH}")
    else:
        print(f">>> 当前已在 {BRANCH} 分支")

    # 4. 设置 remote
    remotes = subprocess.run(
        "git remote", shell=True, capture_output=True, text=True
    ).stdout.strip().split()
    if "origin" in remotes:
        run(f"git remote set-url origin {remote_url}")
    else:
        run(f"git remote add origin {remote_url}")

    # 5. 暂存需要上传的文件（排除数据/输出/缓存）
    targets = [
        "src/", "configs/", "scripts/", "tests/", "docs/",
        "requirements.txt", "pytest.ini", "README.md",
        "QUICKSTART.md", "PROJECT_STRUCTURE.md",
        "PROJECT_ORGANIZATION.md", "run_full_experiment.py",
        "upload_to_github.py", ".gitignore",
        "data/raw/.gitkeep", "data/processed/.gitkeep", "data/cache/.gitkeep",
        "outputs/checkpoints/.gitkeep", "outputs/logs/.gitkeep",
        "outputs/predictions/.gitkeep",
    ]
    run("git add " + " ".join(f'"{t}"' for t in targets))

    # 6. 检查是否有变更需要提交
    status = subprocess.run(
        "git status --porcelain", shell=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        run('git commit -m "Update: sync project code to main branch"')
    else:
        print(">>> 没有新变更，跳过 commit")

    # 7. 推送到 GitHub main 分支
    run(f"git push -u origin {BRANCH} --force")

    print(f"\n[OK] 上传完成！")
    print(f"仓库地址: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")


if __name__ == "__main__":
    main()
