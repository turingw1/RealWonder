import os

# Ensure setuptools is imported before packages that touch distutils indirectly.
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")

try:
    import setuptools  # noqa: F401
except Exception:
    pass
