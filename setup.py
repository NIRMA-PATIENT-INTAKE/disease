from distutils.core import setup

setup(
    name="distool",
    packages=["distool"],
    version="0.1",
    license="MIT",
    description="disease processing tool kit",
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
