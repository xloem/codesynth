from setuptools import find_packages, setup 
import re

# provides for renaming finetuneanon's branch
pkgs = {
    'codesynth': 'codesynth',
    'finetuneanon_transformers_gn_la3_rpb':
        'codesynth/extern/finetuneanon/gpt_neo_localattention3_rp_b/src/transformers'
}

_deps = [
    "Pillow",
    "black==21.4b0",
    "cookiecutter==1.7.2",
    "dataclasses",
    "datasets",
    "deepspeed>=0.3.16",
    "docutils==0.16.0",
    "einops==0.3.0",
    "fairscale>0.3",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flake8>=3.8.3",
    "flax>=0.3.2",
    "fugashi>=1.0",
    "huggingface-hub==0.0.8",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.2.8",
    "jaxlib>=0.1.59",
    "jieba",
    "keras2onnx",
    "nltk",
    "numpy>=1.17",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "packaging",
    "parameterized",
    "protobuf",
    "psutil",
    "pydantic",
    "pytest",
    "pytest-sugar",
    "pytest-xdist",
    "python>=3.6.0",
    "recommonmark",
    "regex!=2019.12.17",
    "requests",
    "rouge-score",
    "sacrebleu>=1.4.12",
    "sacremoses",
    "sagemaker>=2.31.0",
    "scikit-learn",
    "sentencepiece==0.1.91",
    "soundfile",
    "sphinx-copybutton",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme==0.4.3",  # sphinx-rtd-theme==0.5.0 introduced big changes in the style.
    "sphinx==3.2.1",
    "sphinxext-opengraph==0.4.1",
    "starlette",
    "tensorflow-cpu>=2.3",
    "tensorflow>=2.3",
    "timeout-decorator",
    "tokenizers>=0.10.1,<0.11",
    "torch>=1.0",
    "torchaudio",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "uvicorn",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}

setup(
    name='codesynth',
    version='0.0.1',
    packages=[
        *[
            pkg for pkg in pkgs.keys()
        ],
        # rename subpkgs
        *[
            pkg + '.' + subpkg
            for pkg, path in pkgs.items()
            for subpkg in find_packages(path)
        ]
    ],
    package_dir = pkgs,
    install_requires = [
        deps["dataclasses"] + ";python_version<'3.7'",  # dataclasses for Python versions that don't have it
        deps["importlib_metadata"] + ";python_version<'3.8'",  # importlib_metadata for Python versions that don't have it
        deps["einops"], # required for rotary
        deps["filelock"],  # filesystem locks, e.g., to prevent parallel downloads
        deps["huggingface-hub"],
        deps["numpy"],
        deps["packaging"],  # utilities from PyPA to e.g., compare versions
        deps["regex"],  # for OpenAI GPT
        deps["requests"],  # for downloading models over HTTPS
        deps["sacremoses"],  # for XLM
        deps["tokenizers"],
        deps["tqdm"],  # progress bars in model download and training scripts
    ]
)
