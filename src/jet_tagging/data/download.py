import zipfile
import tarfile
import requests
from tqdm import tqdm

from jet_tagging.config import (
    RAW_DIR,
    JETS_RAW_DIR
)

URL = "https://zenodo.org/api/records/3602260/files-archive"


def download_hls4ml(chunk_size=100*1024**2):

    filename = "hls4ml_dataset.zip"
    file_path = JETS_RAW_DIR / filename

    if file_path.exists():
        print('Dataset already exists!')
        return file_path

    with requests.get(URL, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))

        with open(file_path, 'wb') as f, tqdm(
            total=total,
            unit='B',
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

        extracted_files = []

        for member in z.namelist():

            out_path = JETS_RAW_DIR / member

            if out_path.exists():
                continue

            z.extract(member, JETS_RAW_DIR)
            extracted_files.append(out_path)

    return extracted_files


def unpack_tar_files(tar_paths):

    extracted_files = []

    for tar_path in tar_paths:

        with tarfile.open(tar_path, 'r:gz') as tar:

            for member in tar.getmembers():

                out_path = JETS_RAW_DIR / member.name

                if out_path.exists():
                    continue

                tar.extract(member, JETS_RAW_DIR)

                extracted_files.append(out_path)

    return extracted_files


def pipeline_hls4ml(remove_compressed=False):

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = download_hls4ml()

    tar_paths = unpack_zip(zip_path)

    _ = unpack_tar_files(tar_paths)

    if remove_compressed:

        for compressed_file in RAW_DIR.glob("*.zip"):
            compressed_file.unlink()

        for compressed_file in RAW_DIR.glob("*.tar.gz"):
            compressed_file.unlink()

    print("Dataset ready!")

    train_dir = JETS_RAW_DIR / "train"
    return sorted(train_dir.glob("*.h5"))


if __name__ == "__main__":
    files = pipeline_hls4ml()
    for f in files:
        print(f)
