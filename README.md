# Expred Web Application

![Python application](https://github.com/JoshuaGhost/expred_web_app/workflows/Python%20application/badge.svg?branch=JoshuaGhost-action)

## Precondition

To run the web app, you need to install python packages needed according to the requirements.txt by either

```conda install --file requirements.txt```

or

```pip install -r requirements.txt```

## This is yet a prototype!

## How to use

The main entrance of the Web App is the ```app/expred_webapp.py``` file. just run it using python 3.6+. Right now the
port is hard coded as ```8080``` and it can be accessed from localhost ```127.0.0.1``` only.

## How does it look like

The current version of the prototype can only support fact-checking, supported by the ExPred model(s) trained on the
FEVERS dataset. To start with, enter the statement you would like to verify on the index page (e.g. *The ultimate answer
        of Universe is 42*, case insensitive). Then hit the "Verify!" button to fire the query. The app forwards the
query to the Bing custom search and returns (currently) **3** documents retrieved from Wikipedia related to the query.
Current credentials needed by Bing custom search service are mine and hard-coded. There are two such magic strings, one
is the search service credential and the other is the key of the custom configuration file. The ExPred model then
predicts whether the query statement is __SUPPORTED__ or __REFUTED__ by each individual document. The evidence
supporting the prediction are marked with tomato color. If there are more than 4 non-evidence words inbetween, the
words after the 3rd non-evidence word are abbreviated by ```...```.

## Feedback

You can also evaluate the results by selecting whether you feel satisfied or not with the current results. If You choose
'Yes!', current machine-generated explaination is appended to ```data/mgc.csv```. You will be then re-directed to a
'thank you' page with a 'try another statement' button on it, through which you can get back to the index page. Or you
can provide your own classification label and select evidence of the documents supporting/refuting your justification.
Or you can also identify some documents as ```IRRELEVANT``` with your statement, although they may contain some key
words of your query. The annotation contributed by the user are appended to the ```data/ugc.csv``` file.

