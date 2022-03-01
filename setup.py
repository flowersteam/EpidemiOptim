import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EpidemiOptim", # Replace with your own username
    version="0.0.1",
    author="Cédric Colas",
    author_email="cdric.colas@gmail.com",
    description="EpidemiOptim turns epidemiological models into OpenAI Gym environment for the optimization of intervention policies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flowersteam/EpidemiOptim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)