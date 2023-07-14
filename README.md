# TensorFlow minGPT

It's pretty much the same as [Andrej Karpathy's GPT Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) but using TensorFlow instead of PyTorch. Setup and training:

```bash
conda env create -f conda.yml
```

There are two separate commands: one to train a model and another one to generate sequences from the trained model. There are many arguments and optional parameters that can be used from the command line. More:

```bash
python mingpt.py train --help

python mingpt.py generate --help # E.g. python mingpt.py generate "hello"
```
