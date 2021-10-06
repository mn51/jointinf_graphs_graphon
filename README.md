# Graphon-aided Joint Estimation of Multiple Graphs
Joint topology inference of multiple graphs assuming graphs are sampled from the same graphon, a nonparametric random graph model.
The code in this repository contains the methods necessary for reproducing the figures used in the paper [1] submitted to ICASSP 2022.

This repository is organized as follows:
- **jointinf:** contains the joint inference methods with and without augmentations from graphon sampling assumptions.
- **senate_data:** contains CSV files of congressional members, votes, parties, and rollcolls for various congresses.
- **helper.py:** contains useful functions for optimization steps and graphon sampling.
- **fig2a_samenodeset.py**: includes the code for Fig. 2a in the paper.
- **fig2b_samenodeset.py**: includes the code for Fig. 2b in the paper.

For Fig. 2c in the paper, run `fig2b_samenodeset.py` using data from [voteview.com](https://voteview.com/) [2].
```
@misc{lewis2020voteview,
author = {Jeffrey B. Lewis and Keith Poole and Howard Rosenthal and Adam Boche and Aaron Rudkin and Luke Sonnet},
title={Voteview: Congressional Roll-Call Votes Database},
  howpublished = {\url{https://voteview.com/}},
  year=2020,
}
```

[1] M. Navarro and S. Segarra, "Graphon-aided Joint Estimation of Multiple graphs," in *IEEE International Conference on Acoustics, Speech and Signal Processing (to appear)*, 2021.

[2] J. B. Lewis, K. Poole, H. Rosenthal, A. Boche, A. Rudkin, and L. Sonnet, “Voteview: Congressional roll-call votes database.” https://voteview.com/, 2021.
