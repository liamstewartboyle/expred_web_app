import json
import requests
import yaml

with open('credentials.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    subscriptionKey = data['subscriptionKey']
    customConfigId = data['customConfigId']

def get_wiki_page_id(text):
    id_pos = text.index('"wgArticleId":') + len('"wgArticleId":')
    text = text[id_pos:]
    page_id = text[:text.index(',')]
    return page_id


def get_bing_search_urls(res):
    for page in ret['webPages']['value']:
        wiki_ret = requests.get(page['url'])


def bing_wiki_search(search_term):
    bing_search_api = f'https://api.bing.microsoft.com/v7.0/custom/search?q={search_term}&customconfig={customConfigId}'
    search_resp = requests.get(bing_search_api, headers={'Ocp-Apim-Subscription-Key': subscriptionKey})
    res = json.loads(search_resp.text)
    res_urls = [page['url'] for page in res['webPages']['value']]
    return res_urls


def get_wiki_docs(url):
    wiki_ret = requests.get(url)
    page_id = get_wiki_page_id(wiki_ret.text)
    extract_ret = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&pageids={page_id}&formatversion=2&explaintext=1&exsectionformat=plain"
    r = requests.get(extract_ret)
    extract_ret = json.loads(r.text)
    try:
        return extract_ret['query']['pages'][0]['extract']
    except KeyError:
        return ''
    
    
if __name__ == '__main__':
    wiki_urls = bing_wiki_search('bing search')
    docs = [get_wiki_docs(url) for url in wiki_urls]
    print(docs)