from setuptools import find_packages, setup 
import re

# provides for renaming finetuneanon's branch
pkgs = {
    'codesynth': 'codesynth',
    'finetuneanon_transformers_gn_la3_rpb':
        'extern/finetuneanon/gpt_neo_localattention3_rp_b/src/transformers'
}

extras = {
    'client': [ 'requests==2.26.0' ],
    'local': [ 'torch', 'transformers==4.9.2', 'einops==0.3.0' ],
    'server': [ 'aiohttp==3.7.4.post0', 'pjrpc==1.3.0' ],
    'all': 'auto_filled'
}
extras['all'] = [
    extra_dep
    for extra, extra_deps in extras.items()
    for extra_dep in extra_deps
    if extra != 'all'
]

setup(
    name='codesynth',
    version='0.0.1',
    packages=[
        *[
            pkg for pkg in pkgs.keys()
        ],
        # rename subpkgs appropriately
        *[
            pkg + '.' + subpkg
            for pkg, path in pkgs.items()
            for subpkg in find_packages(path)
        ]
    ],
    package_dir = pkgs,
    install_requires = [ ],
    extras_require = extras
)
