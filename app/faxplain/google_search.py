from googlesearch import search
import re

regular_content_name = re.compile('[a-zA-Z0-9_%]*')


def google_wiki_search(query, top=3):
    urls = search(f'{query} site:en.wikipedia.org',
                  lang='en', num=top, start=0, stop=10, pause=2.0)
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