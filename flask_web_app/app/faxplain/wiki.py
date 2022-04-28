import json

import requests


def get_wiki_page_id(text):
    id_pos = text.index('"wgArticleId":') + len('"wgArticleId":')
    text = text[id_pos:]
    page_id = text[:text.index(',')]
    return page_id


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