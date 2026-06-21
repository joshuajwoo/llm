import requests
from bs4 import BeautifulSoup
import trafilatura
import os
import re

def scrape_text(url, output_file, include_math=True):    
    # 1. Send an HTTP GET request to the URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # 2. Use BeautifulSoup to handle math equations BEFORE trafilatura extracts text.
    #    trafilatura doesn't know about LaTeX — it would strip or garble math content.
    #    So we replace math tags with clean LaTeX strings first, then let trafilatura
    #    extract the now-clean HTML.
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 3. Handle math from multiple sources:

    # (a) <math> tags with MathML (Wikipedia and academic sites)
    # These contain <annotation encoding="application/x-tex"> with the LaTeX source.
    for math_tag in soup.find_all('math'):
        if include_math:
            annotation = math_tag.find('annotation', encoding='application/x-tex')
            if annotation and annotation.string:
                latex_str = annotation.string.strip()
                math_tag.replace_with(f' ${latex_str}$ ')
            else:
                math_tag.replace_with('')
        else:
            math_tag.replace_with('')
    
    # (b) MathJax script tags (Stack Exchange, academic blogs, etc.)
    # MathJax stores LaTeX inside <script type="math/tex"> tags.
    for script_tag in soup.find_all('script', type=re.compile(r'math/tex')):
        if include_math:
            latex_str = script_tag.string.strip() if script_tag.string else ''
            if latex_str:
                script_tag.replace_with(f' ${latex_str}$ ')
            else:
                script_tag.replace_with('')
        else:
            script_tag.replace_with('')
    
    # (c) KaTeX spans (Khan Academy, some modern sites)
    # KaTeX renders into <span class="katex"> with the original LaTeX stored in a nested <annotation> element.
    for katex_span in soup.find_all('span', class_='katex'):
        if include_math:
            annotation = katex_span.find('annotation', encoding='application/x-tex')
            if annotation and annotation.string:
                latex_str = annotation.string.strip()
                katex_span.replace_with(f' ${latex_str}$ ')
            else:
                katex_span.replace_with('')
        else:
            katex_span.replace_with('')
    
    # 4. Remove elements that add noise
    # --- wikipedia specific — numbered citations [1] and inline templates [citation needed], [update]
    for sup in soup.find_all('sup', class_='reference'):
        sup.decompose()
    for sup in soup.find_all('sup', class_='noprint'):
        sup.decompose()
    
    # [edit] links next to section headers (MediaWiki sites)
    for edit_span in soup.find_all('span', class_='mw-editsection'):
        edit_span.decompose()
    
    # Remove entire references/notes sections so they don't appear in output.
    # Wikipedia uses <ol class="references"> and sections with id like "References"
    for ref_list in soup.find_all('ol', class_='references'):
        ref_list.decompose()
    for ref_div in soup.find_all('div', class_='reflist'):
        ref_div.decompose()
    # Remove "See also", "References", "External links", "Further reading" sections
    for heading in soup.find_all(['h2', 'h3']):
        heading_text = heading.get_text().strip().lower()
        if heading_text in ['references', 'external links', 'further reading', 'see also', 'notes']:
            # Remove everything from this heading until the next heading of same level
            for sibling in list(heading.find_next_siblings()):
                if sibling.name == heading.name:
                    break
                sibling.decompose()
            heading.decompose()
    # --- end of cleanup of Wikipedia specific tags
    
    # Remove script and style tags
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    
    cleaned_html = str(soup)
    
    text_content = trafilatura.extract(
        cleaned_html,
        include_comments=False, # skip user comments sections
        include_tables=True, # keep data tables
        favor_recall=True, # prefer getting more content over precision
    )
    
    if not text_content:
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.get_text() for p in paragraphs])
    
    # 6. Clean up whitespace
    text_content = re.sub(r'[^\S\n]+', ' ', text_content)       # collapse spaces
    text_content = re.sub(r' +\n', '\n', text_content)          # trim trailing spaces
    text_content = re.sub(r'\n{3,}', '\n\n', text_content)      # max one blank line
    text_content = text_content.strip()
    
    # 7. Save the text to our target file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text_content)
        
    print(f"Saved {len(text_content)} characters to {output_file}")
    return text_content

def scrape_text_to_string(url, include_math=True):
    """Scrape text from a URL and return it as a string without saving to a file."""
    # Reuse scrape_text but write to a temp path, then return the text
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True) as tmp:
        return scrape_text(url, tmp.name, include_math)

def url_to_filename(url):
    from urllib.parse import urlparse
    path = urlparse(url).path
    name = [seg for seg in path.split('/') if seg][-1]
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name).lower()
    return name + '.txt'

if __name__ == '__main__':
    urls = [
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://nypost.com/2026/06/05/us-news/karmelo-anthony-asked-to-leave-15-times-before-fatal-stabbing-witness-says/",
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_path = os.path.join(data_dir, "input.txt")
    delimiter = '<|endoftext|>'

    os.makedirs(data_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out:
        for i, url in enumerate(urls):
            print(f"\nScraping: {url}")
            try:
                # Scrape to a temporary variable instead of a file
                text = scrape_text_to_string(url, include_math=True)
                if text:
                    out.write(text)
                    out.write(f'\n{delimiter}\n')
            except Exception as e:
                print(f"  Error scraping {url}: {e}")

    total_size = os.path.getsize(output_path)
    print(f"\nSaved all content to {output_path} ({total_size:,} bytes)")
