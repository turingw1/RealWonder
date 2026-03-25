# Server Git Workflow

This document describes the recommended Git update workflow on the server for:

- repository path: `~/workspace/Zhengwei/RealWonder`
- conda env: `realwonder`

It is intended for the common case:

- code has been updated on your GitHub repository
- the server should pull the latest version safely
- submodules must stay consistent with the main repository commit
- you want to know when code-only updates are enough and when environment repair is also needed

## 1. Basic Rule

On the server, do not edit the repository state manually unless you know exactly why.

For normal updates, use this order:

1. check for local dirty state
2. pull the main repository
3. sync submodule URLs from `.gitmodules`
4. update submodules recursively
5. run a lightweight environment check
6. only reinstall packages if the update actually changed dependencies

## 2. Standard Update Commands

Run from the server:

```bash
cd ~/workspace/Zhengwei/RealWonder

git status --short --branch
git pull
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

This is the default workflow for almost every update.

## 3. Full Recommended Update Session

Use this version when you want both the code update and a quick health check:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

git status --short --branch
git pull
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive

bash scripts/check_realwonder_env.sh
```

## 4. When `git pull` Is Slow Or Fails

Recommended default on the server: configure Git once for the current user so that all `https://github.com/...` traffic is automatically rewritten to `githubfast`.

```bash
git config --global url."https://githubfast.com/https://github.com/".insteadOf https://github.com/
```

After this, the following commands usually do not need any special mirror syntax:

```bash
git pull
git fetch
git clone --recursive https://github.com/turingw1/RealWonder.git
git submodule sync --recursive
git submodule update --init --recursive
```

This is the simplest setup because:

- the repository remote can stay as the normal GitHub URL
- submodule URLs in `.gitmodules` can also stay as normal GitHub URLs
- Git rewrites them automatically at runtime

To verify the rule:

```bash
git config --global --get-regexp '^url\..*insteadOf$'
```

To remove it later:

```bash
git config --global --unset url."https://githubfast.com/https://github.com/".insteadOf
```

If direct GitHub access is unstable, switch the current repository to `githubfast` temporarily:

```bash
cd ~/workspace/Zhengwei/RealWonder

git remote set-url origin https://githubfast.com/https://github.com/turingw1/RealWonder.git
git pull
```

If needed, switch back afterwards:

```bash
git remote set-url origin https://github.com/turingw1/RealWonder.git
```

If submodule fetches are also slow, sync after `.gitmodules` has been updated:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

If you want all `https://github.com/...` traffic in the current user account to prefer `githubfast`, use:

```bash
git config --global url."https://githubfast.com/https://github.com/".insteadOf https://github.com/
```

This is the preferred server-side setup.

To remove it later:

```bash
git config --global --unset url."https://githubfast.com/https://github.com/".insteadOf
```

## 5. How To Read Submodule Status

Run:

```bash
git submodule status --recursive
```

Interpretation:

- line starts with a space: submodule is checked out at the expected commit
- line starts with `-`: submodule is not initialized
- line starts with `+`: submodule is checked out at a different commit than the main repository expects

If you see `-` or `+`, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 6. Dirty Working Tree Handling

Before `git pull`, always check:

```bash
git status --short --branch
```

If it is clean, continue normally.

If there are local edits:

- if they are your intentional server-only changes, stop and decide whether to keep or stash them
- if they are generated files like `__pycache__`, remove them before updating

Example cleanup:

```bash
find submodules -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```

Then re-check:

```bash
git status --short --branch
```

## 7. When Code Pull Is Enough

You only need code update commands if the change was:

- Python logic fix
- doc update
- script update
- path update
- submodule pointer update with no new Python dependencies

In that case:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

## 8. When You Also Need `pip install`

Run package installation again only if one of these changed:

- `requirements.txt`
- `demo_web/requirements.txt`
- `default.yml`
- `submodules/sam_3d_objects/pyproject.toml`
- `submodules/sam2/setup.py`
- `submodules/sam2/pyproject.toml`
- `submodules/Genesis/pyproject.toml`

Recommended checks:

```bash
git diff --name-only HEAD~1 HEAD
```

If dependency files changed, then run the matching install step again.

Examples:

```bash
python -m pip install -r requirements.txt
python -m pip install -r demo_web/requirements.txt
```

Or for submodules:

```bash
cd submodules/sam2 && python -m pip install -v --no-build-isolation -e .
cd ../Genesis && python -m pip install -v --no-build-isolation -e .
```

## 9. Recommended Checks After Pull

Minimal:

```bash
bash scripts/check_realwonder_env.sh
```

Demo-oriented:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
find /cache/Zhengwei/RealWonder/ckpts -maxdepth 5 | sed -n '1,40p'
find /cache/Zhengwei/RealWonder/wan_models -maxdepth 4 | sed -n '1,40p'
```

## 10. Safe Recovery Commands

If the main repo pulled successfully but submodules are wrong:

```bash
git submodule sync --recursive
git submodule update --init --recursive --force
```

If a submodule URL changed in `.gitmodules` and the server still fetches from the old source:

```bash
git submodule sync --recursive
git config --file .gitmodules --get-regexp 'submodule\..*\.url'
git submodule update --init --recursive
```

## 11. One-Command Update Block

For routine server maintenance:

```bash
git config --global url."https://githubfast.com/https://github.com/".insteadOf https://github.com/

cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

find . -type d -name '__pycache__' -prune -exec rm -rf {} +
git status --short --branch
git pull
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
bash scripts/check_realwonder_env.sh
```

This is the default workflow to use unless you know dependency files changed and need reinstall steps.
