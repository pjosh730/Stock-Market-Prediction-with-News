# Functional Specification
## Background

Daily News have strong effects on Stock Market.
-	Boeing 737 Max crash revelations could cost shareholders $53 billions
-	Fitbit surges 17% after Google egress to buy the company for $2.1 billion 

We want to use Daily News headlines to Predict Stock Market Performance (whether the DJIA increases or not)

## User profile

Stock buyer will use the system. 

The user needs to be able to read, browse website, collect 25 headlines from Reddit, and have basic knowledge on stock market, who need extra information to help them make decision.

## Data sources

Data was collected from [Kaggle](https://www.kaggle.com/lseiyjg/use-news-to-predict-stock-markets) 
- News data was obtained from Reddit WorldNews Channel Top 25 headlines were voted by reddit users for a single date. (Range: 2008-06-08 to 2016-07-01). Each row contains 25 headlines from one day. There are 25 columns of data, and each column contains one of the headlines from that day. 
- Stock data: Dow Jones Industrial Average (DJIA) is used as the label to supervise model training. (Range: 2008-08-08 to 2016-07-01). The DJIA is 1 if it is increases over the day, and 0 if decreased.

## Use cases
### Case1:
1. Objective: User wants to decide whether he needs to buy or sell the stocks today.
2. Interactions: User collects 25 headlines from Reddit and uses them as model input. Then model returns value of 1, which indicates the DJIA will increase during the day. Given prediction and user’s knowledge of stock market, user decides to buy stocks.
### Case2: 
1. Objective: User wants to decide whether he needs to buy or sell the stocks today.
2. Interactions: User collects 25 headlines from Reddit and uses them as model input. Then model returns value of 0, which indicates the DJIA will decrease during the day. Given prediction and user’s knowledge of stock market, user decides to sell stocks.







