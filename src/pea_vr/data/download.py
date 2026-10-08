from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from ..utils import sha256, write_json

YDMS_RECORD = 'https://api.figshare.com/v2/articles/19096823/versions/2'
LONGENOUGH_SOURCE = 'https://github.com/trafnex/video-augmentation#datasets'
LONGENOUGH_FOLDER = 'https://liuonline-my.sharepoint.com/:f:/g/personal/davha914_student_liu_se/ErK6esYd5IdOiuvfLnXK6NoBEdlj579MlXBvG2wkfQEozg?e=sCHtWp'


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_file(url, destination, expected_size=None, md5=None, sha256_value=None, retries=5):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def checked(path):
        return ((expected_size is None or path.stat().st_size == expected_size)
                and (md5 is None or file_digest(path, 'md5') == md5)
                and (sha256_value is None or sha256(path) == sha256_value))
    if destination.exists():
        if not checked(destination):
            raise ValueError(f'Existing file failed checksum: {destination}')
        return destination
    partial = destination.with_name(destination.name + '.part')
    failures = 0
    report_interval = 64 * 1024**2
    next_report = ((partial.stat().st_size if partial.exists() else 0) // report_interval + 1) * report_interval
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset == expected_size:
            break
        if expected_size is not None and offset > expected_size:
            raise ValueError('Partial download is larger than the source file')
        try:
            headers = {'Accept-Encoding': 'identity'}
            end = None
            if expected_size is not None:
                end = min(expected_size - 1, offset + 16 * 1024**2 - 1)
                headers['Range'] = f'bytes={offset}-{end}'
            elif offset:
                headers['Range'] = f'bytes={offset}-'
            request_url = url
            parts = urlsplit(url)
            if parts.hostname == 'ndownloader.figshare.com':
                query = parse_qsl(parts.query) + [('pea_vr_request', f'{offset}_{time.time_ns()}')]
                request_url = urlunsplit(parts._replace(query=urlencode(query)))
            with requests.get(request_url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned a mismatched byte range')
                    if end is not None and response.headers['Content-Range'] != f'bytes {offset}-{end}/{expected_size}':
                        raise ValueError('Server returned an unexpected range length or total')
                    mode = 'ab'
                else:
                    mode, offset = 'wb', 0
                advertised = response.headers.get('Content-Length')
                needed = expected_size - offset if expected_size is not None else int(advertised) if advertised else 0
                if needed and shutil.disk_usage(destination.parent).free < needed + 256 * 1024**2:
                    raise OSError(f'Insufficient disk space for {destination}')
                with partial.open(mode) as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
                            offset += len(block)
                            if offset >= next_report:
                                print(f'[download] {destination.name}: {offset / 1024**2:.0f} MiB', flush=True)
                                next_report = offset + report_interval
            failures = 0
            if expected_size is None or partial.stat().st_size == expected_size:
                break
        except requests.RequestException:
            failures += 1
            if failures >= retries:
                raise
            time.sleep(min(2**failures, 30))
    if not checked(partial):
        raise ValueError(f'Download failed length or checksum verification: {partial}')
    partial.replace(destination)
    return destination


def _safe_path(root, name):
    root = root.resolve()
    if '\\' in name or re.match(r'^[A-Za-z]:', name):
        raise ValueError(f'Unsafe archive entry: {name}')
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Archive entry escapes destination: {name}')
    return target


def extract_archive(archive, destination):
    archive, root = Path(archive), Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            entries = stream.infolist()
            needed = sum(e.file_size for e in entries if not e.is_dir())
            if needed > shutil.disk_usage(root).free:
                raise OSError('Insufficient free space for extracted archive')
            for entry in entries:
                target = _safe_path(root, entry.filename)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symbolic links are not supported')
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(entry) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 1024 * 1024)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as stream:
            for entry in stream:
                target = _safe_path(root, entry.name)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(entry) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 1024 * 1024)
                else:
                    raise ValueError('Archive links and special files are not supported')
    else:
        raise ValueError(f'Unsupported archive: {archive}')
    return root


def download_dataset(dataset, destination, extract=False, url=None, checksum=None):
    if dataset not in ('longenough', 'ydms'):
        raise ValueError('Unknown dataset')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    if dataset == 'ydms' and url is None:
        response = requests.get(YDMS_RECORD, timeout=60)
        response.raise_for_status()
        metadata = response.json()
        write_json(root / 'source_record.json', metadata)
        for file in metadata['files']:
            if Path(file['name']).name != file['name']:
                raise ValueError('Unsafe source filename')
            path = download_file(file['download_url'], root / file['name'], file['size'], file['computed_md5'])
            records.append({'name': file['name'], 'bytes': path.stat().st_size, 'sha256': sha256(path),
                            'source': file['download_url'], 'source_md5': file['computed_md5']})
            if extract and zipfile.is_zipfile(path):
                extract_archive(path, root / 'extracted')
    elif url is not None:
        name = 'longenough.zip' if dataset == 'longenough' else 'dataset.zip'
        path = download_file(url, root / name, sha256_value=checksum)
        records.append({'name': name, 'bytes': path.stat().st_size, 'sha256': sha256(path), 'source': url})
        if extract:
            extract_archive(path, root / 'extracted')
    else:
        from .remote_zip import download_longenough_cohort
        return download_longenough_cohort(root)['files']
    write_json(root / 'download_manifest.json', {'dataset': dataset, 'files': records})
    return records
