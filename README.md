## Sigmoid PyTorch Layer in CUDA

Simple example on how to program in CUDA using Python. Blog post explaining all about it [here](https://www.gfickel.com/jekyll/update/2024/03/13/making-your-gpu-go-brrr-creating-a-cuda-layer-in-pytorch.html)

## How to Run?

Well, we have 2 main files:

- **sigmoid.py**: implements sigmoid forward and backward pass in Pytorch. Also runs gradcheck to confirm that the implementation is OK.
- **numba_cuda.ipynb**: notebook explaining all the steps that I propose to create a CUDA Pytorch layer: implement in Pytorch, then CUDA in Python Numba for debugging, get help of chat-GPT, compile our generated C CUDA in Pyorch and test it out.
- **utils.py**: helper stuff. The main function that we call to compile our C CUDA, **load_model**, is defined there.

## Credits

Jeremy Howard with [this](https://www.youtube.com/watch?v=nOxKexn3iBo) awesome video, and David Oniani for this [Sigmoid Implementation](https://www.youtube.com/watch?v=oxC3T_-_Amw) . Thanks :)
