from setuptools import find_packages, setup 
import re

# provides for renaming finetuneanon's branch
pkgs = {
    'codesynth': 'codesynth',
    'finetuneanon_transformers_gn_la3_rpb':
        'codesynth/extern/finetuneanon/gpt_neo_localattention3_rp_b/src/transformers'
}

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
    install_requires = [ ]
)
