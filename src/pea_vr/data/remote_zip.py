from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import requests

from ..utils import write_json
from .download import LONGENOUGH_FOLDER, _safe_path


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, session, url, size):
        self.session, self.url, self.size = session, url, int(size)
        self.position = 0
        self.cache = {}
        self.etag = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        count = min(count, self.size - self.position)
        if count <= 0:
            return b''
        requested = count
        start, end = self.position, self.position + count - 1
        for (cached_start, cached_end), cached in self.cache.items():
            if cached_start <= start and cached_end >= end:
                self.position += count
                return cached[start - cached_start:end - cached_start + 1]
        end = min(self.size - 1, max(end, start + 65535))
        count = end - start + 1
        key = (start, end)
        if key not in self.cache:
            for attempt in range(5):
                try:
                    response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                                                                  'Accept-Encoding': 'identity'}, timeout=(30, 120), stream=True)
                    with response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise IOError('Source does not support verified HTTP range reads')
                        content_range = response.headers.get('Content-Range', '')
                        if content_range != f'bytes {start}-{end}/{self.size}':
                            raise IOError(f'Unexpected Content-Range: {content_range}')
                        etag = response.headers.get('ETag')
                        if self.etag is not None and etag is not None and etag != self.etag:
                            raise IOError('Remote archive changed during download')
                        self.etag = etag or self.etag
                        data = response.content
                    if len(data) != count:
                        raise IOError('Truncated range response')
                    while self.cache and sum(map(len, self.cache.values())) + count > 32 * 1024**2:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[key] = data
                    break
                except (requests.RequestException, IOError):
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 10))
        data = self.cache[key]
        self.position += requested
        return data[:requested]


def longenough_session():
    session = requests.Session()
    session.get(LONGENOUGH_FOLDER, timeout=(30, 90)).raise_for_status()
    base = 'https://liuonline-my.sharepoint.com/personal/davha914_student_liu_se'
    folder = '/personal/davha914_student_liu_se/Documents/LongEnough-datasets/public'
    endpoint = base + "/_api/web/GetFolderByServerRelativeUrl('" + folder + "')/Files?$select=Name,Length,ServerRelativeUrl,TimeLastModified"
    response = session.get(endpoint, headers={'Accept': 'application/json;odata=nometadata'}, timeout=(30, 90))
    response.raise_for_status()
    files = response.json()['value']
    for item in files:
        item['url'] = 'https://liuonline-my.sharepoint.com' + item['ServerRelativeUrl'] + '?download=1'
    return session, files


def download_longenough_cohort(destination, offsets=(0,), videos=None, progress=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    session, sources = longenough_session()
    source_files = [x for x in sources if x['Name'] in ('LongEnough-variable.zip', 'LongEnough-variable-extended.zip')]
    if len(source_files) != 2:
        raise ValueError('Both variable-bandwidth source archives are required')
    write_json(destination / 'source_record.json', {'dataset': 'longenough', 'public_url': LONGENOUGH_FOLDER,
                                                   'offsets': list(offsets), 'videos': list(videos) if videos is not None else None,
                                                   'archives': source_files})
    import re
    from ..utils import sha256
    manifest_path = destination / 'download_manifest.json'
    manifest = {'dataset': 'longenough', 'files': []}
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    known = {entry['path']: entry for entry in manifest['files']}
    for source in source_files:
        with zipfile.ZipFile(HTTPRangeFile(session, source['url'], source['Length'])) as archive:
            selected = []
            for info in archive.infolist():
                match = re.search(r'(?P<video>\d{4})-(?P<offset>\d{4})-(?P<sample>\d{4})\.(?:qoe\.log|log|bw)$', info.filename)
                if match and int(match['offset']) in offsets and (videos is None or int(match['video']) in videos):
                    selected.append(info)
            if not selected:
                raise ValueError(f'No requested sessions in {source["Name"]}')
            for index, info in enumerate(selected):
                target = _safe_path(destination, info.filename)
                relative = target.relative_to(destination).as_posix()
                if (target.exists() and relative in known and known[relative]['zip_crc32'] == f'{info.CRC:08x}'
                    and known[relative]['bytes'] == info.file_size and sha256(target) == known[relative]['sha256']):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + '.part')
                with archive.open(info) as src, temporary.open('wb') as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)
                temporary.replace(target)
                known[relative] = {'path': relative, 'bytes': info.file_size, 'sha256': sha256(target),
                                   'zip_crc32': f'{info.CRC:08x}', 'archive': source['Name']}
                manifest['files'] = list(known.values())
                write_json(manifest_path, manifest)
                if progress and (index % 25 == 0 or index == len(selected) - 1):
                    print(f'[download] {source["Name"]}: {index + 1}/{len(selected)} files', flush=True)
            archive.fp.cache.clear()
    return manifest
