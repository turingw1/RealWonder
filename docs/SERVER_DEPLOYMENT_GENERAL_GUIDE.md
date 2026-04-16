# General Server Deployment Guide

This document summarizes the practical rules that apply to most projects deployed on a shared GPU server.

Use it when you want a deployment pattern that is:

- reproducible
- quota-aware
- recoverable
- compatible with shared multi-user servers

## 1. Separate Workspace From Cache

Do not treat the code workspace as the place to store everything.

Recommended split:

- workspace: code only
- cache: conda environments, checkpoints, wheels, source mirrors, build caches, temporary data

Typical layout:

```bash
~/workspace/Zhengwei/<project>
/cache/Zhengwei/<project>
```
<user> is always Zhengwei in this case, but you can replace it with your own username if needed.
Inside cache, create explicit directories:

```bash
mkdir -p /cache/<user>/<project>/{conda_envs,hf,torch,torch_extensions,triton,warp,tmp,logs,wheels,src}
mkdir -p /cache/<user>/<project>/{ckpts,models}
```

Why:

- workspace quotas are usually tight
- Git repositories and submodules grow unexpectedly
- build caches and model files can easily consume tens or hundreds of GB

## 2. Put Conda Environments In Cache

Do not default to:

```bash
~/miniconda3/envs/<env_name>
```

Prefer a prefix environment in cache:

```bash
conda create -y -p /cache/<user>/<project>/conda_envs/<env_name> python=3.*
conda activate /cache/<user>/<project>/conda_envs/<env_name>
```

Why:

- it avoids filling the home directory
- it makes cleanup explicit
- it is easier to archive, move, or rebuild

If the environment becomes corrupted, delete the prefix directly and recreate it.

## 3. Use Current-Terminal Environment Variables

On shared servers, prefer per-terminal variables over global shell startup changes.

Typical examples:

```bash
export HF_HOME=/cache/<user>/<project>/hf
export HUGGINGFACE_HUB_CACHE=/cache/<user>/<project>/hf/hub
export TORCH_HOME=/cache/<user>/<project>/torch
export TORCH_EXTENSIONS_DIR=/cache/<user>/<project>/torch_extensions
export TRITON_CACHE_DIR=/cache/<user>/<project>/triton
export WARP_CACHE_DIR=/cache/<user>/<project>/warp
export XDG_CACHE_HOME=/cache/<user>/<project>/tmp
```

Also configure package mirrors in the terminal, not globally:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
```

Why:

- shared servers often use a common Linux account or shared shell defaults
- global changes can silently break other users or future sessions

## 4. Prefer Mirrors, But Keep Official Fallbacks

For Python packages:

- use a fast mirror as the main index
- keep official vendor indexes as extra indexes for framework-specific wheels

For Git:

- if GitHub is unstable, use current-terminal Git overrides

Example:

```bash
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://githubfast.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.version
export GIT_CONFIG_VALUE_1=HTTP/1.1
```

Why:

- many failures are not package problems, but Git transport problems
- `pip` Git dependencies are especially fragile because they clone into `/tmp`

## 5. Do Not Trust One-Shot Install Commands

Commands like:

```bash
pip install -e '.[all]'
```

may appear to succeed while leaving important packages missing.

Safer pattern:

1. install build helpers first
2. install core framework first
3. install problematic CUDA packages explicitly
4. install Git dependencies explicitly
5. only then install umbrella extras

This is especially important for:

- GitHub-based dependencies

## 6. Cache Large Wheels And Source Checkouts

Large wheels should be downloaded once and reused.

Use:

```bash
/cache/<user>/<project>/wheels
```

and install from local wheel paths when possible.

Likewise, for fragile Git dependencies:

- do not rely on `pip` cloning into `/tmp/pip-install-*`
- clone once into:

```bash
/cache/<user>/<project>/src
```

then install from the local checkout:

```bash
python -m pip install -v --no-build-isolation .
```

Why:

- `/tmp` installs are fragile
- network failures are harder to recover from
- persistent source trees are easier to retry and inspect

# optional treatment of CUDA builds as special cases, since they are often the most fragile part of the process, but you should apply this only when normal installation attempts fail with CUDA-related errors
## 7. Treat CUDA Extension Builds As Special Cases

For packages with custom CUDA/C++ builds:

- start with low parallelism
- record the exact CUDA toolchain being used
- prefer explicit architecture settings

Typical safe defaults:

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=1
export NINJA_NUM_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
```

Why:

- multi-job builds can saturate CPU, RAM, or temporary storage
- many failures are not deterministic and disappear when concurrency is lowered

## 8. Be Ready To Switch From Conda CUDA To System CUDA

If you see errors like:

- `fatbinary died due to signal 11`
- strange `nvcc` or `fatbinary` crashes

the project source may be fine, but the active CUDA toolchain may be broken.

In that case, retry using system CUDA:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
unset CUDACXX
unset CUDAHOSTCXX
```

Then verify:

```bash
which nvcc
which fatbinary
nvcc --version
```

Why:

- conda-provided CUDA compilers are convenient but not always stable for heavy extension builds

## 9. Understand That Git Dependencies Are Deployment Risks

If a requirement looks like:

```text
package @ git+https://github.com/org/repo.git@<commit>
```

then you have two independent risks:

1. GitHub network instability
2. source build instability

Best practice:

- clone it manually to cache
- pin the tested commit
- initialize recursive submodules if needed
- install from the local path

This is much more robust than leaving the entire process inside `pip`.

## 10. Always Check Whether a Failure Is Network, Build, or Quota

Do not treat every install failure as a Python dependency issue.

Usually failures fall into three buckets:

### Network

Examples:

- HTTP/2 framing errors
- clone checkout failed
- pip download stalls at very low speed

### Build

Examples:

- missing headers
- `fatbinary` crash
- `ninja: build stopped: subcommand failed`

### Quota / disk

Examples:

- `Disk quota exceeded`
- `project block limit reached`
- `unable to checkout working tree`

The fix depends completely on which category it belongs to.

## 11. Watch Disk Usage Early

Check both workspace and cache:

```bash
du -sh ~/workspace/<project>
du -sh ~/workspace/<project>/.git
du -sh ~/workspace/<project>/.git/modules/* 2>/dev/null
df -h ~/workspace /cache
```

Common hidden storage offenders:

- `.git/modules/...`
- large submodule histories
- `torch_extensions`
- `triton` and `warp` caches
- `pip` temporary source trees
- model checkpoints accidentally stored in workspace

If disk is full, stop installing first. Free space first.

## 12. Keep Runtime Paths Stable With Symlinks

If code expects repo-relative paths, but files must live in cache, use symlinks:

```bash
ln -sfn /cache/<user>/<project>/ckpts ~/workspace/<project>/ckpts
ln -sfn /cache/<user>/<project>/models ~/workspace/<project>/models
```

Why:

- you preserve code expectations
- you keep large files out of Git and workspace quotas

## 13. Validate In Layers

Do not jump directly to the full demo or application.

Validate in order:

1. Python and torch
2. critical libraries
3. model checkpoints
4. offline pipeline
5. interactive or web UI

This is the fastest debugging loop.

## 14. Make Recovery Cheap

A good server deployment is not only installable, but recoverable.

Keep these operations easy:

- recreate environment prefix
- purge caches
- reclone a single Git dependency
- relink checkpoints from cache
- re-run only one layer of validation

That is why environments, wheels, source trees, and models should all have separate homes.

## 15. Recommended Generic Deployment Checklist

For any new project on a shared GPU server:

1. create cache directories first
2. create conda prefix under cache
3. configure terminal-only cache and mirror variables
4. clone code into workspace
5. sync submodules explicitly
6. install packaging helpers
7. install torch first
8. install fragile CUDA/Git dependencies explicitly
9. place checkpoints under cache and link them back
10. validate in layers
11. monitor disk usage during the whole process
12. document every non-obvious workaround you needed

## 16. Short Rule Summary

If you only remember a few rules, remember these:

- code in workspace, data and environments in cache
- do not trust one-shot install commands for complex projects
- if `pip` pulls from GitHub, be ready to clone manually and install locally
- if CUDA build tools crash, try system CUDA
- if disk is full, stop and fix space before retrying
- if a server is shared, prefer current-terminal configuration only
