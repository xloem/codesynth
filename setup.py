from setuptools import setup

setup(
    name='codesynth',
    version='0.0.1',
    packages=['codesynth', 'finetuneanon_transformers_gn_la3_rpb'],
    package_dir = {
        'finetuneanon_transformers_gn_la3_rpb':
            'codesynth/extern/finetuneanon/gpt_neo_localattention3_rp_b/src/transformers'
    },
    install_requires=[]
)
