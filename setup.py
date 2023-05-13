import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mechaFIL",
    version="1.8",
    author="Maria Silva, Tom Mellan, Kiran Karra, Vik Kalghatgi, Nicola",
    author_email="misilva73@gmail.com, t.mellan@imperial.ac.uk, kiran.karra@gmail.com, vik@protocol.ai",
    description="API wrapper for mechaFIL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/protocol/mechafil-webserver",
        "Source": "https://github.com/protocol/mechafil-webserver",
    },
    packages=["mechafil"],
    install_requires=["pandas==1.5.3", "mechaFIL==1.8", "flask", "streamlit", "watchdog", "matplotlib", "mpld3"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
