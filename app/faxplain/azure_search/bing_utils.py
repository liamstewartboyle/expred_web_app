import json
import requests
import yaml

from wiki import get_wiki_docs

with open('azure_search/credentials.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    subscriptionKey = data['subscriptionKey']
    customConfigId = data['customConfigId']


def get_bing_search_urls(res):
    for page in res['webPages']['value']:
        wiki_ret = requests.get(page['url'])


def bing_wiki_search(search_term):
    bing_search_api = f'https://api.bing.microsoft.com/v7.0/custom/search?q={search_term}&customconfig={customConfigId}'
    search_resp = requests.get(bing_search_api, headers={'Ocp-Apim-Subscription-Key': subscriptionKey})
    res = json.loads(search_resp.text)
    res_urls = [page['url'] for page in res['webPages']['value']]
    return res_urls


if __name__ == '__main__':
    wiki_urls = bing_wiki_search('bing search')
    docs = [get_wiki_docs(url) for url in wiki_urls]
    print(docs)
