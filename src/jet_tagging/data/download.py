from pathlib import Path
import zipfile
import tarfile
import requests
from tqdm import tqdm

from jet_tagging.config import (
    DATA_DIR,
    RAW_DIR, 
)

RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://zenodo.org/api/records/3602260/files-archive"

def download_hls4ml(chunk_size=1024 * 1024):

    filename = "hls4ml_dataset.zip"
    file_path = RAW_DIR / filename

    if file_path.exists():
        print('Dataset already exists!')
        return file_path

    with requests.get(URL, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))

        with open(file_path, 'wb') as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc='Downloading dataset'
        ) as pbar:

            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return file_path

def unpack_zip(zip_path):

    with zipfile.ZipFile(zip_path) as z:

        members = z.namelist()

        for m in members:

            out_path = RAW_DIR / m

            if out_path.exists():
                continue

            z.extract(m, RAW_DIR)

    return members

def unpack_tar_files():

    extracted = []

    for tar_path in RAW_DIR.glob("*.tar.gz"):

        with tarfile.open(tar_path, 'r:gz') as tar:

            for member in tar.getmembers():

                out_path = RAW_DIR / member.name

                if out_path.exists():
                    continue

                tar.extract(member, RAW)

                extracted.append(out_path)

    return extracted

def pipeline_hls4ml(remove_compressed=True):

    zip_path = download_hls4ml()

    unpack_zip(zip_path)

    unpack_tar_files()

    if remove_compressed:
        for c in RAW_DIR.glob("*.zip"):
            c.unlink()

        for c in RAW.glob("*.tar.gz"):
            c.unlink()

    print('Dataset ready!')

    return sorted((RAW / "train").glob("*.h5"))

if __name__ == "__main__": 
    files = pipeline_hls4ml()