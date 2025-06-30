On top of the `/siamese-net/train-test.ipynb` file, you can also find how I initiated the environment and installed libraries. I got a .yml output of my environment just in case, and the content is available in the `spec-file.yml` file.

Here you can see the environment building setup as well:

```
conda create --name sig_verif_1 python=3.8.19
```
```
conda activate sig_verif_1
```
```
conda install -c conda-forge numpy=1.24.4 pandas=1.1.4 matplotlib=3.3.3 pillow=8.1.1
```

More info for getting started with PyTorch, installation, and the available editions is found at https://pytorch.org/get-started/locally.

```
conda install -c conda-forge notebook
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
