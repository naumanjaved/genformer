import setuptools
import os
import sys
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(version_path)
from version import __version__
    
setuptools.setup(
    name=__version__,
    version="0.1.0",
    author="N Javed",
    author_email="javed@broadinstitute.org",
    description="predicting rna seq from atac + sequence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BernsteinLab/genformer",
    project_urls={
        "Bug Tracker": "https://github.com/BernsteinLab/genformer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        'tensorflow',
        'numpy',
        'tensorflow_addons',
        'wandb'
    ],
)
