# coding: utf-8
"""
**input/output**: ``entities.json``  
Parses the dump (should be downloaded first, or auto-downloaded by the preprocessor below), gathers images and assign them to the relevant entity given its common categories (retrieved in ``wiki.py commons rest``).
Note that the wikicode is parsed very lazily and might need a second run depending on your application, e.g. templates are not expanded...

Usage: wikidump.py <subset> [--max_threads=<max_threads>]

Options:
--max_threads=<n>                Maximum number of threads to use for concurrent image downloading [default: 4].
"""

import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from docopt import docopt
import json
import pandas as pd

from meerqat.data.loading import DATA_ROOT_PATH
from meerqat.data.wiki import VALID_ENCODING

import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

NAMESPACE = {"mw": "http://www.mediawiki.org/xml/export-0.10/"}


def download_single_file(link, download_dir):
    filename = link.split("/")[-1]
    file_path = download_dir / filename
    if file_path.exists():
        print(f"{filename} đã tồn tại, bỏ qua.")
        return file_path
    print(f"Đang tải {filename} ...")
    with requests.get(link, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Đã tải xong {filename}")
    return file_path


def download_dump_files(root_url, download_dir, max_threads=1, limit=None):
    print(f"Accessing dump page: {root_url}")
    response = requests.get(root_url)
    if response.status_code != 200:
        print("Không thể truy cập trang dump.")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    pattern = re.compile(r"^commonswiki-latest-pages-articles[1-6]\.xml-.*\.bz2$")
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and pattern.match(href):
            links.append(urljoin(root_url, href))
    if limit:
        links = links[:limit]
    print(f"Tìm thấy {len(links)} file dump cần tải về.")
    download_dir.mkdir(exist_ok=True)
    downloaded_files = []
    for i in range(0, len(links), max_threads):
        group = links[i:i + max_threads]
        print(f"Đang xử lý nhóm file: {group}")
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download_single_file, link, download_dir): link for link in group}
            for future in as_completed(futures):
                try:
                    file_path = future.result()
                    downloaded_files.append(file_path)
                except Exception as e:
                    print(f"Lỗi tải file {futures[future]}: {e}")
    return downloaded_files


def parse_file(path):
    if path.suffix == ".bz2":
        with bz2.open(path, "rb") as file:
            tree = ET.parse(file)
    else:
        tree = ET.parse(path)
    return tree


def find(element, tag, namespace=NAMESPACE):
    if element is None:
        return None
    return element.find(tag, namespace)


def find_text(element, tag, namespace=NAMESPACE):
    result = find(element, tag, namespace)
    if result is None:
        return None
    return result.text


def get_field(wikitext, image, field):
    result = re.findall(rf"{field}=\s*(.+)\n", wikitext)
    if result:
        image[field.lower()] = result[0]
    return result


def process_article(article, entities, entity_categories):
    for page in article:
        title = find_text(page, "mw:title")
        # keep only files with valid encoding
        if title is None or not title.startswith("File:") or title.split('.')[-1].lower() not in VALID_ENCODING:
            continue

        revision = find(page, "mw:revision")
        if revision is None:
            continue
        wikitext = find_text(revision, "mw:text")
        if wikitext is None:
            continue

        # find categories
        categories = set()
        for internal_link in re.findall(r"\[\[(.+)\]\]", wikitext):
            if internal_link.lower().startswith("category:"):
                name = internal_link.find("|")
                if name >= 0:
                    internal_link = internal_link[:name]
                categories.add("C" + internal_link[1:])
        # is there any entity with these categories?
        if not (categories & entity_categories):
            continue

        image = {"categories": list(categories),
                 "timestamp": find_text(revision, "mw:timestamp")}
        contributor = find(revision, "mw:contributor")
        image["username"] = find_text(contributor, "mw:username")
        for field in ["Date", "Author"]:
            get_field(wikitext, image, field)

        description = re.search(r"description\s*=\s*(.+)", wikitext, flags=re.IGNORECASE|re.DOTALL|re.MULTILINE)
        if description is not None:
            description = description.group(1)
            i_new_field = description.find("\n|")
            if i_new_field >= 0:
                description = description[:i_new_field]
        image["description"] = description

        for license_match in re.finditer(r"{{int:license-header}}\s*=+", wikitext):
            license_ = re.findall("{{.+}}", wikitext[license_match.end():])
            if license_:
                image["license"] = license_[0]
            break

        # find entities with appropriate categories and save the image
        for entity in entities.values():
            if entity["n_questions"] < 1:
                continue
            if entity.get("categories", {}).keys() & categories:
                entity.setdefault("images", {})
                entity["images"][title] = image

    return entities


def process_articles(dump_path, entities):
    categories = {category for entity in entities.values() if entity["n_questions"] > 0
                           for category in entity.get("categories", {})}
    articles_path = list(dump_path.glob(r"commonswiki-latest-pages-articles[0-9]*"))
    for article_path in tqdm(articles_path, desc="Processing articles"):
        article = parse_file(article_path).getroot()
        process_article(article, entities, categories)
    return entities


if __name__ == "__main__":
    args = docopt(__doc__)
    subset = args['<subset>']
    max_threads = int(args['--max_threads'])

    root_dump_url = "https://dumps.wikimedia.org/commonswiki/latest/"
    dump_dir = DATA_ROOT_PATH / "commonswiki"

    # downloaded_files = download_dump_files(root_dump_url, dump_dir, max_threads=max_threads, limit=None)
    # print(f"Number of dump files: {len(downloaded_files)}")

    # load entities
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    path = subset_path / "entities.json"
    with open(path, 'r') as file:
        entities = json.load(file)

    process_articles(dump_dir, entities)

    # save output
    with open(path, 'w') as file:
        json.dump(entities, file)

    print(f"Successfully saved output to {path}")

    n_images = [len(entity.get('images', {})) for entity in entities.values()]
    print(f"Gathered images from {len(entities)} entities:\n{pd.DataFrame(n_images).describe()}")
