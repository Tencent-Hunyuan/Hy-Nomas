#!/usr/bin/env python3
"""Bind six uploaded GitHub video attachments to both READMEs.

Usage: python tools/set_video_urls.py --urls /path/to/video_urls.json
The JSON maps case_01 through case_06 to their actual GitHub attachment URLs.
This command verifies the uploaded MP4 resources and edits the local READMEs.
It does not upload or publish anything.
Use --check to reject a README that still points at local videos or file links.
"""
import argparse
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN CASE VIDEOS -->'
END = '<!-- END CASE VIDEOS -->'
PATTERN = re.compile(r'https://github\.com/user-attachments/assets/[0-9a-fA-F-]{36}')
KEYS = tuple(f'case_{i:02d}' for i in range(1, 7))


def case_table(media):
    """Two columns, with a standalone media paragraph in each table cell."""
    rows = []
    for start in range(0, len(KEYS), 2):
        labels = ''.join(f'<th>Case {i+1:02d}</th>' for i in range(start, start+2))
        cells = '\n'.join(f'<td width="50%">\n\n{media[KEYS[i]]}\n\n</td>'
                          for i in range(start, start+2))
        rows.append(f'<tr>{labels}</tr>\n<tr>\n{cells}\n</tr>')
    return START + '\n\n<table>\n' + '\n'.join(rows) + '\n</table>\n\n' + END


def video_block(urls):
    if set(urls) != set(KEYS) or any(not PATTERN.fullmatch(urls[k]) for k in KEYS):
        raise ValueError('Provide six actual github.com/user-attachments/assets URLs, keyed case_01 to case_06.')
    if len(set(urls.values())) != 6:
        raise ValueError('Each case must have its own video attachment.')
    return case_table(urls)


def verify_video(url):
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-63', 'User-Agent': 'IDM-video-check'})
    with urllib.request.urlopen(request, timeout=45) as response:
        header = response.read(64)
    if b'ftyp' not in header:
        raise ValueError('Attachment does not return an accessible MP4 video: ' + url)


def validate_readmes(root=ROOT, online=False):
    shared = None
    for name in ('README.md', 'README_CN.md'):
        text = (root / name).read_text()
        block = text.split(START, 1)[1].split(END, 1)[0]
        urls = {key: url for key, url in zip(KEYS, PATTERN.findall(block))}
        expected = video_block(urls)[len(START):-len(END)]
        if block != expected:
            raise ValueError(f'{name}: use the two-column case table with standalone GitHub attachment URLs in its cells.')
        if shared is not None and urls != shared:
            raise ValueError('Both README languages must use the same six video attachments.')
        shared = urls
    if online:
        for url in shared.values():
            verify_video(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--urls', type=Path)
    choice.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        validate_readmes(online=True)
        print('Six inline video attachments are accessible in both README tables.')
        return
    urls = json.loads(args.urls.read_text())
    block = video_block(urls)
    for url in urls.values():
        verify_video(url)
    updates = {}
    for name in ('README.md', 'README_CN.md'):
        text = (ROOT / name).read_text()
        before, remainder = text.split(START, 1)
        _, after = remainder.split(END, 1)
        updates[name] = before + block + after
    for name, text in updates.items():
        (ROOT / name).write_text(text)
    validate_readmes()
    print('Both README tables now use inline GitHub video attachments.')


if __name__ == '__main__':
    main()
