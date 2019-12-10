# Component Specification

## Software components
Data Manager: 
Natural Language ToolKit and Scikit-Learn.
The system will use Natural Language ToolKit and Scikit-Learn to preprocessing text data. Those two will help transfer the dataset into useable vectors.

Visualization: 
Word Cloud and Matplotlib.
Use matplotlib and word cloud libraries to visualize text data. They can build plots with size of each words indicate their frequency and importance. Meanwhile the prediction will be given.

Machine Learning Moduel:
Random Forest and Naive Bayes.
Those two moduels will provide a accurancy for our prediction t support the correctness.

## Interactions to accomplish use cases
The system will do a stock price prediction job；

The inputs will be the daily top 25 news headlines. Data manager will process the input data and make them readable by python language;

The outputs will be a visualization of stock price trend show whether its increasing or decreasing;

Final result: With those input, the system will start analysis and provide a prediction on the following day stock price with “0” or “1” to represent its trending. 
