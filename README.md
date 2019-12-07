# Using reddit news headlines to predict stock market trend of the day 

## Goal: 
 - Use the reddit news headlines to predict stock market trend (increase or decrease) 
 - Main dependent variable of interest: Dow Jones Industrial Average (DJIA) 
 - Model input: top 25 daily news headlines from Reddit 
 
## User specification:
- This packaged is intended for individuals who are interested in the daily stock market trend and want to understand how much news of that day can influence market trennd 

## How to use this package: 
- Install jupyter notebooks on your operating system (follow instructions on: https://jupyter.org/install)
- Install Git Bash shell (if you are using a Windows system) 
- Type the following command into your terminal/Git bash shell: 
   $ git clone https://github.com/pjosh730/Stock-Market-Prediction-with-News
 - Initiate a new jupyter notebook
 - Download the first 25 news headlines on Reddit (https://www.reddit.com/r/news/)
 - Load a python module of interest: ex. NaiveBayesModel.py 
 - Make predictions (1 means increase in DJIA, 0 means unchanged or decreased in DJIA) 
 
 ## Software dependencies: 
 Below are the necessary software and packages to run models in this directory:
 
 Python version: 3.7 (https://www.python.org)
 
 Python packages needed:
 
 - Matplotlib 
 
 - Numpy (version: 1.16.4)
 
 - Pandas (version: 0.24.2)
 
 - NLTK (version: 3.4.4)
 
 - Scikit-learn (version: 0.21.2)
 
 - Wordcloud (version: 1.5.0)
 
 - Pickle (Version: 0.7.5)
 
License information:
Stock-Market-Prediction-with-News is licensed under a BSD 2-clause “Simplified” License. The objective behind this choice of licensing is to make the content reproducible and make it useful for as many people as possible. We want to maximize the two-way collaborations with minimum restrictions, so that developers of other projects can easily utilize, patch, improve, and cite this code. For detailed description of the contents of license please refer to License
 
 ## Project structure 
 The package is organized as follows:
 
 Stock-Market-Prediction-with-News Home (master)
|     .gitignore
|     LICENSE
|     README.md
|     setup.py
|  
|----- doc
|     |      CSE_583_Technology_Review.pdf
|     |      Component Speficification.md
|     |      Functional Specification.md
|     |      code_backup_ori.ipynb
|     |      homework5.docx
|     |
|----- Stock-Market-Preiction-with-News
|     |   __init__.py
|     |   NaiveBayesModel.py
|     |   random_forest.py
|     |   .DS_Store
|     |   __pycache__
|     |
|     |----- tests
|     |      |    __init__.py
|     |      |    README.md
|     |      |    interactive_plots_unittests.ipynb
|     |      |    test_data_processing.py
|     |      |    test_network_tools.py
|     |      |    test_plotting.py
|     |      |    test_sensitivity_tools.py
|     |
|     |----- Data
|     |      |    .DS_Store
|     |      |    Combined_News_DJIA.csv
|     |      |    dailynews.csv
|     |
