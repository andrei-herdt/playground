import subprocess


def run_command(command):
    return subprocess.check_output(command, shell=True).decode().strip()


def get_current_commit_hash():
    return run_command("git rev-parse --short HEAD")


def get_remote_url():
    return run_command("git remote get-url origin")


def transform_github_url(remote_url):
    if remote_url.startswith("git@github.com:"):
        return remote_url.replace("git@github.com:", "https://github.com/").replace(
            ".git", ""
        )
    elif remote_url.startswith("https://"):
        return remote_url.replace(".git", "")
    else:
        raise ValueError("Unsupported URL format")


def get_commit_url(commit_hash, web_url):
    return f"{web_url}/commit/{commit_hash}"


def print_url():
    commit_hash = get_current_commit_hash()
    remote_url = get_remote_url()
    web_url = transform_github_url(remote_url)
    commit_url = get_commit_url(commit_hash, web_url)
    print(commit_url)


if __name__ == "__main__":
    print_url()
