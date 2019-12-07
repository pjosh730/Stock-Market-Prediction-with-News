import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Stock Market Prediction with News", # Replace with your own username
    version="0.0.1",
    author="Shaohang Hao, Weikun Hu, Ji Peng and Ruiyu Zeng",
    author_email="pjosh730@gmail.com",
    description="A package can be used to predict stock market with daily top25 News",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pjosh730/Stock-Market-Prediction-with-News",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
