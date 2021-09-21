# Storm Tracking Analyzer
STAnalyzer

<!-- badges: start -->

[![Software License](https://img.shields.io/badge/license-MIT-green)](https://github.com/helvecioneto/stanalyzer/blob/master/LICENSE)
[![Software Life Cycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://www.tidyverse.org/lifecycle/#maturing)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/helvecioneto/stanalyzer/blob/main/docs/Example.ipynb)

<!-- badges: end -->

## Install Dependency
## Anaconda Python 3

$ conda env create --file stanalyzer.yml

$ conda activate stanalyzer

## Examples
Global variables for translate_data.py

PATH = '/home/helvecioneto/01_IARA/RADAR/iara_beta_v6/output/S201409070000_E201409100000_VDBZc_T20_L5_SPLTTrue_MERGTrue_TCORTrue_PCORFalse.zip'<br></br>
DATA_PATH = '/home/helvecioneto/SINAPSE_01/DADOS/sbandradar/'<br></br>
VAR_NAME = 'DBZc'<br></br>
LEVEL = 5<br></br>
THRESHOLD = [20,35,40]<br></br>
OUTPUT = '../output/'<br></br>
NC_OUTPUT = '../output/data/'<br></br>
OUTPUT_FILE = '../output/output_file_tracked'<br></br>

### Usage

python translate_data.py

## Jupyter notebook

<a href="https://nbviewer.jupyter.org/github/helvecioneto/stanalyzer/blob/main/docs/Example.ipynb" target="_blank">Example for EDA tracked files</a>

## Contact info

