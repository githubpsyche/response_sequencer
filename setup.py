import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="response_sequencer",
    version="0.1",
    author="Jordan Gunn",
    author_email="gunnjordanb@gmail.com",
    description="techniques for automatic preprocessing of complex free response data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/githubpsyche/response_sequencer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],
)