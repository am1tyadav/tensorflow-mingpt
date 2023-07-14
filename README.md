# TensorFlow minGPT

It's pretty much the same as [Andrej Karpathy's GPT Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) but using TensorFlow instead of PyTorch.

I don't have a GPU, so my approach is to use [SkyPilot](https://github.com/skypilot-org/skypilot) to spin up a GPU instance. The following command spins up an instance and starts training the model.

```bash
sky launch mingpt.yml --down -i 1
```
