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

## 关于deploy到google app engine的方法:

现在用的是一个monkey patch，用以规避app engine的10分钟build timeout

方法是：先把程序部署到gcloud run上，然后在app engine里面deploy一个静态页面，重定向到run的那个实例，因为在www demo paper里面写的那个url是[https://faxplain.appspot.com/](https://faxplain.appspot.com/) ，这是一个app engine分配的url。

### gcloud run的使用方法：

首先切换目录到app路径下，然后在这个下面执行以下命令，该命令使用cloudbuild.yaml文件种列出的命令一条条打包/上传/部署docker container到gcloud run上：

```gcloud commit --config cloudbuild.yaml .```
```gcloud build submit --config cloudbuild.yaml .```

~~用以下命令来把当前目录打包成docker镜像，并上传到gcloud run上进行部署。需要mem：4GB（这个是从run的service里面选的，一个service相当于一个虚拟机）。运行下面这个命令之后会有prompt问service名称，写“default”，区域选12，public available~~
~~```gcloud commit beta run deploy --source . --project faxplain --platform managed```~~
~~在当前目录中应该有一个名为```Procfile```的文件，这个文件是docker image要执行的命令，在这里用```web:```开头~~
~~```web: FLASK_APP=expred_webapp.py python3 -m flask run --host=0.0.0.0 --port=8080```~~

### 用gae进行重定向：

在gae里面，只deploy一个静态页面（redirect/redirect.html），文件已经有了，到时候就根据run的url更改redirect目标就好。在redirect文件夹下有一个app.yaml文件，用法：

```gcloud app deploy app.yaml --project faxplain```

## 收集数据：

资料源自[这里](https://cloud.google.com/sdk/gcloud/reference/compute/copy-files)

使用类似如下命令：

```gcloud compute copy-files example-instance:~/REMOTE-DIR  ~/LOCAL-DIR --zone=us-central1-a```

而云端镜像存在[这里](https://console.cloud.google.com/gcr/images/faxplain?project=faxplain&folder&organizationId)


```gsutil rsync data gs://mybucket/data```
5. 设置ci过程
6. 从本地进行gcloud deployment操作
7. 固定gcloud run的网址，最好是不用gae转发