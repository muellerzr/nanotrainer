# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup


extras = {}
extras["quality"] = [
    "black ~= 23.1",  # hf-doc-builder has a hidden dependency on `black`
    "hf-doc-builder >= 0.3.0",
    "ruff ~= 0.2.1",
]

setup(
    name="nanotrainer",
    version="0.0.1.dev",
    description="nanoTrainer, a minimal Hugging Face Trainer implementation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="The HuggingFace team",
    author_email="zach.mueller@huggingface.co",
    url="https://github.com/muellerzr/nanotrainer",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.8.0",
    install_requires=[
        "accelerate",
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.10.0",
        "huggingface_hub",
        "safetensors>=0.3.1",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# Add in based on accelerate's `setup.py` when ready for a release
