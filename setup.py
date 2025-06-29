"""
Twinkle Eval 套件安裝配置
高效且準確的 AI 模型評測工具
"""

import os

from setuptools import find_packages, setup


# 讀取 README 檔案作為長描述
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# 讀取需求檔案
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


setup(
    name="twinkle-eval",
    version="1.1.0",
    author="Twinkle AI Team",
    description="🌟 高效且準確的 AI 模型評測工具",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ai-twinkle/Eval",
    project_urls={
        "Bug Tracker": "https://github.com/ai-twinkle/Eval/issues",
        "Documentation": "https://github.com/ai-twinkle/Eval#readme",
        "Source Code": "https://github.com/ai-twinkle/Eval",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "twinkle-eval=twinkle_eval.cli:main",
        ],
    },
    package_data={
        "twinkle_eval": [
            "*.yaml",
            "*.yml",
            "config.template.yaml",
        ],
    },
    include_package_data=True,
    keywords=[
        "ai",
        "llm",
        "evaluation",
        "benchmark",
        "machine-learning",
        "natural-language-processing",
        "artificial-intelligence",
        "testing",
    ],
    zip_safe=False,
)
