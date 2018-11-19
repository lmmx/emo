from emoji_dict import emoji_urls
from config import UCD_XML
from pathlib import Path
from subprocess import check_output
from bs4 import BeautifulSoup as bs
from sys import stderr
import requests

def DeprecatedProcessName(n):
    p, ext = n.split('.')
    p = p.lower()
    if p.startswith('emoji_icon_-_'):
        p = p[13:]
    if p.startswith('emoji_icon_'):
        p = p[11:]
    e = p.find('_emoji_icon')
    if e > -1:
        p = p[:e] + p[e+11:]
    j = p.find('_emoji')
    if j > -1:
        p = p[:j] + p[j+6:]
    if p.endswith('_large'):
        p = p[:-6]
        if len(p.split('_')[-1]) == 36:
            p = p[:-37]
    if p.endswith('_ios10'):
        p = p[:-6]
    if p.endswith('_png'):
        p = p[:-4]
    #if p.endswith('_1') or p.endswith('_2') or p.endswith('_3'):
    #    p = p[:-2]
    p = p.replace('_-_', '_')
    p = p.replace('-', '_')
    # Fix typo
    if p == 'grinmacing': p = 'grimacing'
    for x in ['alien', 'ghost', 'heart_eyes', 'sleeping', 'wink']:
        if p.startswith(x):
            if p.endswith('_2'):
                p = p[:-2]
    # There are still 2 duplicates: "happy", "sick_2"
    return f'{p}.{ext}'

def NameEmoji(e):
    """
    Get hex code, strip the '0x' prefix, and uppercase.
    """
    assert UCD_XML.is_file()
    enc = hex(ord(e))[2:].upper()
    c = check_output(["grep",f'cp="{enc}"',UCD_XML]).strip()
    title = bs(c, 'lxml').char.attrs['na'].lower().split(' ')
    if 'face' in title:
        title.remove('face')
    n = '_'.join(title)
    return n

def DownloadFile(fname, url):
    """
    Name png file according to emoji's unicode character,
    save locally.
    """
    local_fname = Path(__file__).parent / 'img' / fname
    if not local_fname.is_file():
        print(f'Creating file: {fname}', file=stderr)
        r = requests.get(url)
        with open(local_fname, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded {emoji} ---> {fname}", file=stderr)
    else:
        print(f"Skipped {emoji} ({fname} already exists)", file=stderr)
    return

for emoji, url in emoji_urls.items():
    # download emoji_url, strip its name
    if len(emoji) > 1:
        if hex(ord(emoji[1])) == '0xfe0f':
            # remove this "variation selector"
            emoji = emoji[0]
    ext = url.split('.')[-1]
    fname = NameEmoji(emoji) + f'.{ext}'
    DownloadFile(fname, url)
