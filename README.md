# codesynth

A small draftpouch of code generation synergy.

Contains a handful of models and api interfaces that can generate source code.
This is done based on up to 2k words of preceding context.

Additionally, there is an rpc server and client to use them remotely.

This can now be installed as a python package.

To run locally or as an rpc server with the included model:
```
git-lfs install --skip-repo
git submodule update --init # downloads genji model

pip3 install .[all]

export TRANSFORMERS_MODELS="$(pwd)/extern"

codesynth-server # begin listening for rpc requests
```

Or to install the library for use as a lightweight rpc client:
```
pip3 install .[client]
```

Usage example:
```
import codesynth

generator = codesynth.genji() # or codesynth.rpc_client() after running codesynth-server

print('Have some generated code:')

print(generator('def hello_world():'))
```
