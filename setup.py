from distutils.core import setup

setup(
    name="distool",  # How you named your package folder (MyLib)
    packages=["distool"],  # Chose the same as "name"
    version="0.1",  # Start with a small number and increase it with every change you make
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="disease processing tool kit",  # Give a short description about your library
    author="NIRMA Team of ITMO University",  # Type in your name
    author_email="egorovmichil9@gmail.com",  # Type in your E-Mail
    url="https://github.com/nirma-patient-intake/disease/",  # Provide either the link to your github or to your website
    download_url="https://github.com/NIRMA-PATIENT-INTAKE/disease/archive/refs/heads/main.zip",  # I explain this later on
    keywords=[
        "NLP",
        "Disease",
        "Health Condition",
    ],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        "scikit-learn",
        "beautifulsoup4",
        "negspacy",
        "numpy",
        "spacy",
        "negspacy",
        "attrs",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",  # Again, pick a license
        "Programming Language :: Python :: 3",  # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
