# PaLM

This is an implementation of the work published by google called **PaLM: Scaling Language Modeling with Pathways**.

```
@misc{https://doi.org/10.48550/arxiv.2204.02311,
  doi = {10.48550/ARXIV.2204.02311}, 
  url = {https://arxiv.org/abs/2204.02311},  
  author = {Chowdhery, Aakanksha and Narang, Sharan and Devlin, Jacob and Bosma, Maarten and Mishra, Gaurav and Roberts, Adam and Barham, Paul and Chung, Hyung Won and Sutton, Charles and Gehrmann, Sebastian and Schuh, Parker and Shi, Kensen and Tsvyashchenko, Sasha and Maynez, Joshua and Rao, Abhishek and Barnes, Parker and Tay, Yi and Shazeer, Noam and Prabhakaran, Vinodkumar and Reif, Emily and Du, Nan and Hutchinson, Ben and Pope, Reiner and Bradbury, James and Austin, Jacob and Isard, Michael and Gur-Ari, Guy and Yin, Pengcheng and Duke, Toju and Levskaya, Anselm and Ghemawat, Sanjay and Dev, Sunipa and Michalewski, Henryk and Garcia, Xavier and Misra, Vedant and Robinson, Kevin and Fedus, Liam and Zhou, Denny and Ippolito, Daphne and Luan, David and Lim, Hyeontaek and Zoph, Barret and Spiridonov, Alexander and Sepassi, Ryan and Dohan, David and Agrawal, Shivani and Omernick, Mark and Dai, Andrew M. and Pillai, Thanumalayan Sankaranarayana and Pellat, Marie and Lewkowycz, Aitor and Moreira, Erica and Child, Rewon and Polozov, Oleksandr and Lee, Katherine and Zhou, Zongwei and Wang, Xuezhi and Saeta, Brennan and Diaz, Mark and Firat, Orhan and Catasta, Michele and Wei, Jason and Meier-Hellstern, Kathy and Eck, Douglas and Dean, Jeff and Petrov, Slav and Fiedel, Noah},  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},  
  title = {PaLM: Scaling Language Modeling with Pathways},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Dependencies

- `torch`
- `torchvision`
- `tensorboardX`
- `tqdm`
- `einops`
- `pyyaml`
- `ipython`
- `pytest`
- `pytest-xdist`

## Usage

### Training a PaLM model from scratch:

#### Download and prepare the dataset, then train the PaLM model: 
```bash
$ python3 download.py --path data/ --dataset wikitext-103 --combine True --tokenize True --split False --process True --batch_size 256 --seq_len 256 --vocab_size 50000 --vocab_file data/vocab.txt --n_workers 2 --download_data False --download_models False --download_scripts False --download_pretrained True
$ python3 train.py
```
