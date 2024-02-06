#!/usr/bin/env python3
import hashlib
import os

from typing import Optional
from urllib.request import urlopen, Request
from pathlib import Path
from zipfile import ZipFile

REPO_BASE_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = REPO_BASE_DIR / "_data"
BEGINNER_DATA_DIR = REPO_BASE_DIR / "beginner_source" / "data"
INTERMEDIATE_DATA_DIR = REPO_BASE_DIR / "intermediate_source" / "data"
ADVANCED_DATA_DIR = REPO_BASE_DIR / "advanced_source" / "data"
PROTOTYPE_DATA_DIR = REPO_BASE_DIR / "prototype_source" / "data"
FILES_TO_RUN = os.getenv("FILES_TO_RUN")


def size_fmt(nbytes: int) -> str:
    """Returns a formatted file size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    return str(nbytes) + " bytes"


def download_url_to_file(url: str,
                         dst: Optional[str] = None,
                         prefix: Optional[Path] = None,
                         sha256: Optional[str] = None) -> Path:
    dst = dst if dst is not None else Path(url).name
    dst = dst if prefix is None else str(prefix / dst)
    if Path(dst).exists():
        print(f"Skip downloading {url} as {dst} already exists")
        return Path(dst)
    file_size = None
    u = urlopen(Request(url, headers={"User-Agent": "tutorials.downloader"}))
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    sha256_sum = hashlib.sha256()
    with open(dst, "wb") as f:
        while True:
            buffer = u.read(32768)
            if len(buffer) == 0:
                break
            sha256_sum.update(buffer)
            f.write(buffer)
    digest = sha256_sum.hexdigest()
    if sha256 is not None and sha256 != digest:
        Path(dst).unlink()
        raise RuntimeError(f"Downloaded {url} has unexpected sha256sum {digest} should be {sha256}")
    print(f"Downloaded {url} sha256sum={digest} size={size_fmt(file_size)}")
    return Path(dst)


def unzip(archive: Path, tgt_dir: Path) -> None:
    with ZipFile(str(archive), "r") as zip_ref:
        zip_ref.extractall(str(tgt_dir))


def download_hymenoptera_data():
    # transfer learning tutorial data
    z = download_url_to_file("https://download.pytorch.org/tutorial/hymenoptera_data.zip",
                             prefix=DATA_DIR,
                             sha256="fbc41b31d544714d18dd1230b1e2b455e1557766e13e67f9f5a7a23af7c02209",
                             )
    unzip(z, BEGINNER_DATA_DIR)


def download_nlp_data() -> None:
    # nlp tutorial data
    z = download_url_to_file("https://download.pytorch.org/tutorial/data.zip",
                             prefix=DATA_DIR,
                             sha256="fb317e80248faeb62dc25ef3390ae24ca34b94e276bbc5141fd8862c2200bff5",
                             )
    # This will unzip all files in data.zip to intermediate_source/data/ folder
    unzip(z, INTERMEDIATE_DATA_DIR.parent)


def download_dcgan_data() -> None:
    # Download dataset for beginner_source/dcgan_faces_tutorial.py
    z = download_url_to_file("https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip",
                             prefix=DATA_DIR,
                             sha256="46fb89443c578308acf364d7d379fe1b9efb793042c0af734b6112e4fd3a8c74",
                             )
    unzip(z, BEGINNER_DATA_DIR / "celeba")


def download_lenet_mnist() -> None:
    # Download model for beginner_source/fgsm_tutorial.py
    download_url_to_file("https://docs.google.com/uc?export=download&id=1HJV2nUHJqclXQ8flKvcWmjZ-OU5DGatl",
                         prefix=BEGINNER_DATA_DIR,
                         dst="lenet_mnist_model.pth",
                         sha256="cb5f8e578aef96d5c1a2cc5695e1aa9bbf4d0fe00d25760eeebaaac6ebc2edcb",
                         )

def download_gpu_quantization_torchao() -> None:
    # Download SAM model checkpoint for prototype_source/gpu_quantization_torchao_tutorial.py
    download_url_to_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                         prefix=PROTOTYPE_DATA_DIR,
                         dst="sam_vit_h_4b8939.pth",
                         sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
                         )

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    BEGINNER_DATA_DIR.mkdir(exist_ok=True)
    ADVANCED_DATA_DIR.mkdir(exist_ok=True)
    INTERMEDIATE_DATA_DIR.mkdir(exist_ok=True)
    PROTOTYPE_DATA_DIR.mkdir(exist_ok=True)

    if FILES_TO_RUN is None or "transfer_learning_tutorial" in FILES_TO_RUN:
        download_hymenoptera_data()
    nlp_tutorials = ["seq2seq_translation_tutorial", "char_rnn_classification_tutorial", "char_rnn_generation_tutorial"]
    if FILES_TO_RUN is None or any(x in FILES_TO_RUN for x in nlp_tutorials):
        download_nlp_data()
    if FILES_TO_RUN is None or "dcgan_faces_tutorial" in FILES_TO_RUN:
        download_dcgan_data()
    if FILES_TO_RUN is None or "fgsm_tutorial" in FILES_TO_RUN:
        download_lenet_mnist()
    if FILES_TO_RUN is None or "gpu_quantization_torchao_tutorial" in FILES_TO_RUN:
        download_gpu_quantization_torchao()

if __name__ == "__main__":
    main()
