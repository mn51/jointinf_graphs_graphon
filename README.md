# Joint Network Topology Inference via a Shared Graphon Model
Joint topology inference of multiple graphs assuming graphs are sampled from the same graphon, a nonparametric random graph model.
The code in this repository contains the methods necessary for reproducing the figures used in papers [1] published in ICASSP 2022 and [2] submitted to IEEE Transactions on Signal Processing.

This repository is organized as follows:
- **jointinf:** contains the joint inference methods with and without augmentations from graphon sampling assumptions.
- **senate_data:** contains CSV files of congressional members, votes, parties, and rollcolls for various congresses.
- **methods.py:** contains the joint inference methods with and without augmentations from graphon sampling assumptions.
- **methods_weighted.py:** contains the same joint inference methods as in `methods.py` modified for weighted graphs. 
- **synthetic_test.py**: includes the code for estimating multiple networks sampled from the same graphon using synthetic stationary graph signals.
- **synthetic_weighted_test.py**: includes the code for estimating multiple *weighted* networks sampled from the same graphon using synthetic stationary graph signals.
- **fig2b_samenodeset.py**: includes the code for Fig. 2b in the paper [1].

For Fig. 2c in [1], run methods in `methods.py` using data from [voteview.com](https://voteview.com/) [3].
```
@misc{lewis2020voteview,
author = {Jeffrey B. Lewis and Keith Poole and Howard Rosenthal and Adam Boche and Aaron Rudkin and Luke Sonnet},
title={Voteview: Congressional Roll-Call Votes Database},
  howpublished = {\url{https://voteview.com/}},
  year=2020,
}
```

For Fig. 6 in [2], run methods in `methods.py` using data from the Allen Brain Atlas [4].
```
@article{lein2007genome,
  title={Genome-wide atlas of gene expression in the adult mouse brain},
  author={Ed S. Lein and others},
  journal={Nature},
  volume={445},
  number={7124},
  pages={168--176},
  year={2007},
  publisher={Nature Publishing Group}
}
```

[1] M. Navarro and S. Segarra, "Graphon-aided Joint Estimation of Multiple graphs," in *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022.

[2] M. Navarro and S. Segarra, "Joint Network Topology Inference via a Shared Graphon Model," *arXiv preprint*, *arXiv:2209.08223*, 2022.

[3] J. B. Lewis, K. Poole, H. Rosenthal, A. Boche, A. Rudkin, and L. Sonnet, “Voteview: Congressional roll-call votes database.” https://voteview.com/, 2021.

[4] E. S. Lein et al., "Genome-wide atlas of gene expression in the adult mouse brain," in *Nature*, 2007.
