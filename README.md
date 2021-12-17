# Fall detection for motiongrams - FYS-STK4155 Project 3

This repository contains the code for fall detection classification using motiongrams and some code for doing bias-variance analysis.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages

```bash
pip install -r requirements.txt
```

## Usage

```
usage: src/run.py [-h] [-m] [-c] [-b] [-p] [-a]

To run the bias-variance tradeoff experiment

optional arguments:
  -h, --help          show this help message and exit
  -m, --models        To run test of different models to evaluate their performance
  -c, --cnn           To run the pytorch and tensorflow cnn analyses
  -b, --biasvariance  To test the bias-variance tradeoff for three different methods
  -p, --plot          To make the plots and output useful results
  -a, --all           To run all the analyzes
```

## Data

Motiongrams obtained from videos from:
http://fenix.univ.rzeszow.pl/mkepski/ds/uf.html

30 falls and 40 adl-s (activities of caily life)

## Structure

```
.
├── data                - contains the data
├── output              - contains the output
│  ├── data             - contains the data
│  └── plots            - contains the plots
├── requirements.txt    - contains the requirements
└── src                 - contains the source code
   ├── analysis.py      - contains the analysis code
   ├── bias_variance.py - contains the bias-variance code
   ├── cnn.py           - contains the cnn code
   ├── config.py        - contains the configuration variables
   ├── data.py          - contains the data code
   ├── plot.py          - contains the plot code
   ├── preview.py       - contains the preview code
   ├── regression.py    - contains the regression code
   └── run.py           - contains the run code
```
