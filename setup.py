from distutils.core import setup

from setuptools import find_packages

setup(
    name="distool",
    description="Disease processing tool kit in Russian",
    packages=find_packages(),
    package_data={"distool": ["data/symptoms.json"]},
    include_package_data=True,
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    version="0.1.8.1",
    license="MIT",
    author="NIRMA Team of ITMO University",
    author_email="egorovmichil9@gmail.com",
    url="https://github.com/nirma-patient-intake/disease/",
    download_url="https://github.com/NIRMA-PATIENT-INTAKE/disease/archive/refs/tags/distool.tar.gz",
    keywords=[
        "NLP",
        "Disease",
        "Health Condition",
    ],
    install_requires=[
        "scikit-learn",
        "beautifulsoup4",
        "negspacy",
        "numpy",
        "spacy",
        "negspacy",
        "attrs",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
