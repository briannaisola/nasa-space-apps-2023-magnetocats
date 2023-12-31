# Terrestrial-Space Weather Integrated Forecasting Tool (T-SWIFT)
**It's me, hi. I'm T-SWIFT**. Welcome to the landing page!


T-SWIFT is a neural network pipeline for predicitng geomagnetic storms created part of the [2023 NASA Space Apps Challenge](https://www.spaceappschallenge.org/) hackathon.

This model was developed by team [Magnetocats](https://www.spaceappschallenge.org/2023/find-a-team/magnetocats/), featuring six PhD students from the[ University of New Hampshire Space Space Science Center](https://eos.unh.edu/space-science-center): Mayowa Adewuyi, Mike Coughlan, James Edmond, Brianna Isola, Raman Mukundan and Neha Srivastava.

## The Challenge 

[Develop the Oracle of DSCOVR](https://www.spaceappschallenge.org/2023/challenges/develop-the-oracle-of-dscovr/): When operating reliably, the National Oceanic and Atmospheric Administration’s (NOAA’s) space weather station, the Deep Space Climate Observatory (DSCOVR), can measure the strength and speed of the solar wind in space, which enables us to predict geomagnetic storms that can severely impact important systems like GPS and electrical power grids on Earth. DSCOVR, however, continues to operate past its expected lifetime and produces occasional faults that may themselves be indicators of space weather. Your challenge is to use the "raw" data from DSCOVR—faults and all—to predict geomagnetic storms on Earth.

See:
* [NASA Space Apps Challenge 2023 Homepage](https://www.spaceappschallenge.org/)
* [Team Magnetocats Homepage](https://www.spaceappschallenge.org/2023/find-a-team/magnetocats/)
* [The DSCOVR Mission - NASA](https://science.nasa.gov/mission/dscovr/)

### Running T-Swift
_Are you ready for it?_
1. Clone this repository in your directory of choice.
2. Unzip the data files in the `data/dscovr/` directory. Leave the uncompressed files in that directory.
3. To train the Gap-Filler model, which creates synthetic DSCOVR data, run `gap_filling.py`.
4. To make forecasts of the Hp30 index using the Forecaster model, run `ANN_forecaster.py`.

## Directory & File Descriptions
* `data/dscovr/`; loation of raw DSCOVR datafiles
* `figures`; example figures and images
* `data-overview.py`;  Jupyter Notebook with examples for processing and plotting data


