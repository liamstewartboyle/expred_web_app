from googlesearch import search
import re
import yaml

import requests
import json

regular_content_name = re.compile('[a-zA-Z0-9_%\(\)]*')


def purify_urls(urls, top):
    # google search returns different chapters in the same wiki page (differentiated by ids) as
    # different results. Here we need to remove such duplicated links
    rets = []
    for url in urls:
        trimmed_url = url.split('#')[0]
        site, category, page_name = trimmed_url.split('/')[-3:]
        if site != 'en.wikipedia.org' or category != 'wiki' \
                or not regular_content_name.fullmatch(page_name):  # The last condition rules out Files:
            continue
        if not rets or trimmed_url != rets[-1]:
            rets.append(trimmed_url)
        if len(rets) == top:
            break
    return rets


def google_wiki_search(query, top=3):
    urls = search(f'{query} site:en.wikipedia.org',
                  lang='en', num=3 * top, start=0, stop=3 * top, pause=2.0)
    rets = purify_urls(urls, top)
    return rets


def google_rest_api_search(raw_query: str, top=10):
    with open('credentials.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        api_key = data['google_api_key']
        cx = data['google_api_cx']
    query = raw_query.replace(' ', '+')
    headers = {'Accept': 'application/json'}
    url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&' \
          f'q={query}&' \
          f'cx={cx}&' \
          f'num={top * 3}'
    response = requests.get(url, headers=headers)
    res = json.loads(response.text)
    if 'items' not in res:
        return []
    res_urls = [item['link'] for item in res['items']]
    res = purify_urls(res_urls, top)
    return res


def bing_wiki_search(search_term):
    with open('credentials.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        subscriptionKey = data['bingSubscriptionKey']
        customConfigId = data['bingCustomConfigId']
    bing_search_api = f'https://api.bing.microsoft.com/v7.0/custom/search?q={search_term}&customconfig={customConfigId}'
    search_resp = requests.get(bing_search_api, headers={'Ocp-Apim-Subscription-Key': subscriptionKey})
    res = json.loads(search_resp.text)
    res_urls = [page['url'] for page in res['webPages']['value']]
    return res_urls


if __name__ == '__main__':
    # wiki_urls = bing_wiki_search('bing search')
    # docs = [get_wiki_docs(url) for url in wiki_urls]
    wiki_urls = google_rest_api_search('this is a test')
    print(wiki_urls)
