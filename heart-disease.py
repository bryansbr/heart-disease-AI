# -*- coding: utf-8 -*-
"""
File name: heart-disease.ipynb
Creation date: 03/11/2020
Modification date: 04/11/2020
Authors: Bryan Steven Biojó     - 1629366
         Julián Andrés Castaño  - 1625743
		 Juan Sebastián Saldaña - 1623447
		 Juan Pablo Rendón      - 1623049

"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
url = "https://raw.githubusercontent.com/bryansbr/heart-disease-AI/main/heart.csv"
data = pd.read_csv(url)
data.head()

"""Test
colors = {"satisfiable": "Blue", "unsatisfiable": "Red"}
instances_color = data.clause_result.map(colors)
"""


