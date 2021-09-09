import argparse
from bs4 import BeautifulSoup
import os
import random
import requests
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import unicodedata
from multiprocessing import Pool
import json


SRE_MATCH_TYPE = type(re.match("", ""))


def parse_index(body):
    """Parse wikipedia article reference pages to retrieve article links."""
    dates = [(m.start(0), int(m.group(1))) for m in re.finditer(r"<span class.*? id.*?>(\d{4})<\/span>", str(body))]
    articles = [(m.start(0), m.group(1), m.group(2)) for m in re.finditer(r"<li>\d+.*?\<a .*?href=\"\/wiki\/(.*?)\" title=\"(.*?)\"", str(body))]
    articles_list = []
    for (art_idx, art_ref, art_title) in articles:
        for (date_idx, date) in dates:
            if art_idx > date_idx:
                art_date = date
        articles_list.append({'ref': art_ref, 'date': art_date, 'title': art_title})
    return articles_list


def clean_xml(xml, replacements = {"&gt;" : ">", "&lt;" : "<"}):
    repl_str = xml
    for char in replacements:
        repl_str = repl_str.replace(char, replacements[char])
    return repl_str


def remove_html_tags(text):
    if type(text) == SRE_MATCH_TYPE:
        text = text.group(2)
    text = re.sub(r"<(.*?)(?: .*?)?>(.*?)<\/(\1)>", remove_html_tags, text)
    return text


def clean_paragraph(paragraph):
    paragraph = unicodedata.normalize("NFKD", paragraph)
    paragraph = remove_html_tags(paragraph)
    paragraph = paragraph.strip()
    return paragraph


def get_article_content(link):
    """Return a wikipedia article content given its link."""
    try:
        page = requests.get("https://fr.wikipedia.org/w/api.php?"
                            "format=xml"
                            "&action=query"
                            "&prop=extracts"
                            "&titles={}"
                            "&redirects=true".format(link))
        page_xml = clean_xml(page.text)
        soup = BeautifulSoup(page_xml, 'xml')
        extract = soup.find('extract')
        p = extract.find_all('p')
        p = [clean_paragraph(pp.text) for pp in p]
        return link, p
    except:
        print("couldn't parse {}".format(link))
        return None, None


def save_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        for art in tqdm(dataset):
            json.dump(art, f)
            f.write('\n')
            # for p in paragraphs[art_idx]:
            #     f.write('{}\n'.format(p))
            # f.write('\n')


def save_title(titles, file_path):
    with open(file_path, 'w') as f:
        for k, v in titles.items():
            f.write(k+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--good_articles', action='store_true', help="name of the dataset")
    parser.add_argument('--year_min', default=2003, type=int, help="minimum year")
    parser.add_argument('--year_max', default=2018, type=int, help="maximum year")
    parser.add_argument('--output_dir', required=True, help="output dir to save data")
    parser.add_argument('--n_procs', required=True, type=int, help="number of processes to parallelize download")
    args = parser.parse_args()

    urls = {
        'featured_articles': 'https://fr.wikipedia.org/wiki/'
                             'Wikip%C3%A9dia:Articles_de_qualit%C3%A9/Justification_de_leur_promotion',
        'good_articles': 'https://fr.wikipedia.org/wiki/'
                         'Wikip%C3%A9dia:Bons_articles/Justification_de_leur_promotion'
    }

    featured_links = [urls['featured_articles'] + '/' + str(y) for y in range(args.year_min, args.year_max)]
    featured_links.insert(0, urls['featured_articles'])

    print('Will parse {} page(s):'.format(len(featured_links)))
    for l in featured_links:
        print("\t{}".format(l))

    articles = []

    for fl in tqdm(featured_links, desc='Extract articles titles'):
        page = requests.get(fl)
        soup = BeautifulSoup(page.content, 'html.parser')
        html = list(soup.children)[2]
        body = list(html.children)[3]
        articles.extend(parse_index(body))

    print("Listed {} featured articles".format(len(articles)))

    with Pool(args.n_procs) as p:
        paragraphs = list(tqdm(p.imap(get_article_content, [a['ref'] for a in articles]), 
            total=len(articles), desc='Parse featured articles'))

    for p in paragraphs:
        for idx in range(len(articles)):
            if articles[idx]['ref'] == p[0]:
                articles[idx]['content'] = p[1]

    articles = [a for a in articles if 'content' in a]

    X_train, X_test = train_test_split(articles, test_size=120)
    X_test, X_valid = train_test_split(X_test, test_size=60)

    if args.good_articles:
        good_links = [urls['good_articles'] + '/' + y for y in range(args.year_min, args.year_max)]
        good_links.insert(0, urls['good_articles'])

        print('Will parse {} page(s):'.format(len(good_links)))
        for l in good_links:
            print("\t{}".format(l))

        good_articles = []
        for fl in tqdm(good_links):
            page = requests.get(fl)
            soup = BeautifulSoup(page.content, 'html.parser')
            html = list(soup.children)[2]
            body = list(html.children)[3]
            good_articles.extend(parse_index(body))

        print("Listed {} good articles".format(len(good_articles)))

        with Pool(args.n_procs) as p:
            paragraphs = list(tqdm(p.imap(get_article_content, [a['ref'] for a in good_articles]), 
                total=len(good_articles), desc='Parse good articles'))

        for p in paragraphs:
            for idx in range(len(good_articles)):
                if good_articles[idx]['ref'] == p[0]:
                    good_articles[idx]['content'] = p[1]

        good_articles = [a for a in good_articles if 'content' in a]

        X_train = good_articles + X_train
        random.shuffle(X_train)
    
    save_dataset(X_train, os.path.join(args.output_dir, 'wiki.train.jsonl'))
    save_dataset(X_valid, os.path.join(args.output_dir, 'wiki.valid.jsonl'))
    save_dataset(X_test, os.path.join(args.output_dir, 'wiki.test.jsonl'))
