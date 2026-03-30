# Server Git Workflow

This document describes the recommended Git workflow on the server for a multi-person collaboration setup.

Repository path:

- `~/workspace/Zhengwei/RealWonder`

Conda environment:

- `realwonder`

This workflow now assumes:

- `main` is the stable integration branch
- `zhengwei-dev` is the primary daily development branch
- daily work should not happen directly on `main`
- stable results should move to `main` through PR and merge
- collaborators should also work through branches and PRs, not direct pushes to `main`

## 1. Branch Roles

Use the following branch policy as the default:

- `main`
  - stable
  - demoable
  - deployable
  - only receives reviewed and validated work

- `zhengwei-dev`
  - main daily development branch
  - default branch for server-side research and engineering work
  - accepts regular integration from feature branches

- `feature/*`
  - short-lived branches for focused tasks
  - merge into `zhengwei-dev` first

Recommended merge direction:

1. `feature/* -> zhengwei-dev`
2. `zhengwei-dev -> main`

## 2. Basic Rule

On the server:

- do not use `main` for normal development
- do not run experiments on `main`
- do not push unfinished work directly to `main`

The server should normally stay on:

- `zhengwei-dev`

Only switch to `main` when you explicitly need to:

- inspect the stable state
- validate the released branch
- fast-forward after a stable merge has already landed

## 3. Standard Daily Update Workflow

This is now the default start-of-day workflow.

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

git status --short --branch
git checkout zhengwei-dev
git pull origin zhengwei-dev
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive

bash scripts/check_realwonder_env.sh
```

This is the normal command sequence for almost all daily work.

## 4. Daily Development Pipeline

The stable day-to-day pipeline is:

1. update `zhengwei-dev`
2. create a short-lived branch if the task is isolated enough
3. implement and test
4. push branch to GitHub
5. open PR into `zhengwei-dev`
6. after `zhengwei-dev` is stable, open PR from `zhengwei-dev` into `main`

Recommended branch naming:

- `feature/<topic>`
- `fix/<topic>`
- `exp/<topic>`

Example:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

git checkout zhengwei-dev
git pull origin zhengwei-dev
git checkout -b feature/two-step-distill-ablation
```

After work is complete:

```bash
git add <files>
git commit -m "Add two-step distillation ablation"
git push -u origin feature/two-step-distill-ablation
```

Then open a PR:

- base: `zhengwei-dev`
- compare: `feature/two-step-distill-ablation`

## 5. Stable Promotion Workflow

When `zhengwei-dev` reaches a stable checkpoint:

1. make sure the branch is green enough to demo or reproduce
2. open a PR from `zhengwei-dev` into `main`
3. merge only after validation

That means `main` becomes a release-like branch, not a scratch branch.

Recommended promotion checklist before PR to `main`:

- core demo path still runs
- no broken imports
- no missing checkpoints caused by path changes
- dependency changes are documented
- key experiment logs or outputs are saved

## 6. How To Sync Stable `main` Back Into `zhengwei-dev`

If `main` receives a stable merge and you want the server to continue daily work on top of it:

```bash
cd ~/workspace/Zhengwei/RealWonder
conda activate realwonder

git checkout main
git pull origin main

git checkout zhengwei-dev
git pull origin zhengwei-dev
git merge main
```

If there are no conflicts, push:

```bash
git push origin zhengwei-dev
```

Use this when `main` has become the newest validated baseline.

## 7. How Collaborators Should Work

The expected collaboration model is:

- no one should treat `main` as a daily work branch
- collaborators should branch from `zhengwei-dev` or another agreed development branch
- they should open PRs into `zhengwei-dev`
- only validated integrated work should move onward to `main`

This keeps:

- `main` stable
- `zhengwei-dev` active
- collaboration conflicts localized before release

## 8. When To Work Directly On `zhengwei-dev`

Working directly on `zhengwei-dev` is acceptable only when:

- the change is small
- the risk is low
- you still intend to push and review it before promotion to `main`

Even then, for larger or riskier work, prefer:

- `feature/* -> zhengwei-dev`

## 9. Dirty Working Tree Handling

Before pulling or switching branches, always check:

```bash
git status --short --branch
```

If the tree is clean, continue.

If there are local edits:

- if they are intentional, commit them or stash them
- if they are generated files, remove them first

Example cleanup:

```bash
find submodules -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```

Then re-check:

```bash
git status --short --branch
```

## 10. Network-Safe Pull Workflow

If GitHub is slow on the server, prefer shell-local rewrite instead of global git config:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
```

Then run normal commands:

```bash
git pull
git fetch
git submodule sync --recursive
git submodule update --init --recursive
```

Unset afterwards if needed:

```bash
unset GIT_CONFIG_COUNT
unset GIT_CONFIG_KEY_0
unset GIT_CONFIG_VALUE_0
```

## 11. Submodule Update Workflow

After every branch update, especially after pulling teammate changes:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

Interpretation of submodule status:

- leading space: correct commit checked out
- leading `-`: not initialized
- leading `+`: checked out at a different commit than expected

If you see `-` or `+`, run the sync and update commands again.

## 12. When Code Pull Is Enough

A normal branch update is enough if the changes are only:

- Python logic
- docs
- scripts
- path fixes
- submodule pointer changes without new dependencies

Typical workflow:

```bash
git checkout zhengwei-dev
git pull origin zhengwei-dev
git submodule sync --recursive
git submodule update --init --recursive
```

## 13. When You Also Need Reinstallation

Reinstall only if dependency files changed, for example:

- `requirements.txt`
- `demo_web/requirements.txt`
- `default.yml`
- `submodules/sam_3d_objects/pyproject.toml`
- `submodules/sam2/pyproject.toml`
- `submodules/Genesis/pyproject.toml`

Check recent changed files:

```bash
git diff --name-only HEAD~1 HEAD
```

If dependency files changed, rerun only the relevant install step.

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

## 14. Recommended Checks After Every Meaningful Pull

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

## 15. Recovery Commands

If the main repo updated but submodules are wrong:

```bash
git submodule sync --recursive
git submodule update --init --recursive --force
```

If you accidentally ended up on `main` for daily work:

```bash
git checkout zhengwei-dev
git pull origin zhengwei-dev
```

If `zhengwei-dev` is behind `main` after a stable release:

```bash
git checkout main
git pull origin main
git checkout zhengwei-dev
git merge main
git push origin zhengwei-dev
```

## 16. Bottom Line

The server should now be operated under this default assumption:

- `main` is stable
- `zhengwei-dev` is the daily working branch
- feature work lands in `zhengwei-dev` first
- stable work reaches `main` only through PR and merge

That is the collaboration-first workflow going forward.

