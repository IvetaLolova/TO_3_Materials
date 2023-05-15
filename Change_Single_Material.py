
# Vzdálenost rozprostření gausů se počítá jak v řádku tak v sloupci.
# Možná je potřeba si pohrát s tím zaokrouhlením.


# ----------------------------------------------
# Script for create Gaussian network
# 2019
# ----------------------------------------------
# ----------------------------------------------
# Import
# ----------------------------------------------
import win32com.client
import numpy as np
# import pymxwl
import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow
# from PyQt5 import uic
import statistics

from math import pi, cos, sin, exp, pow, sqrt
from shutil import copyfile
import os
import re
import shutil
import math
import time
from random import random
# import pandas
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# from colorspacious import cspace_converter
from collections import OrderedDict


Surrounding_materials = [3,3,1,2]
print('Surrounding_materials len', len(Surrounding_materials))

# Material_Iron      = 1
# Material_Air       = 2
# Material_Aluminium = 3
# Material_PM        = 4

Material_Count=[0,0,0,0,0]          #Materials_Count = [None, Iron, Air, Aluminium, PM]
for i in range(len(Surrounding_materials)):
    Material_Count[i]= Surrounding_materials.count(i)

# Material_Count=[0,0,0,0,0]          #Materials_Count = [None, Iron, Air, Aluminium, PM]
# for i in range(len(Surrounding_materials)):
#     Material_Count[Surrounding_materials[i]]= Material_Count[Surrounding_materials[i]] + 1

print('Material_Count', Material_Count)

print('Material_Count', max(Material_Count))
print('Surrounding_materials_sort', Surrounding_materials.sort())
print('Surrounding_materials', Surrounding_materials)

print('Material_Count_MAx_index', Material_Count.index(max(Material_Count)))
print('(len(Surrounding_materials)/2)', (len(Surrounding_materials)/2))

Final_Material_After_Change = 0

if max(Material_Count) > (len(Surrounding_materials)/2):
    # if maximal count of surrounding materials is bigger then half of surrounding elements,
    # than change single_element material to that material of majority of surrounding elements
    Final_Material_After_Change = Material_Count.index(max(Material_Count))
elif max(Material_Count) <= (len(Surrounding_materials)/2):
    # if maximal count of surrounding materials is lower or even to half of surrounding elements,
    # than change single_element material to that material as follows starting with Air as prefered material:
    if Material_Count[2] == max(Material_Count):            # Prefer Air instead of Iron/Aluminum/PM
        Final_Material_After_Change = 2
    elif Material_Count[1] == max(Material_Count):          # Prefer Iron instead of Aluminum/PM
        Final_Material_After_Change = 1
    elif Material_Count[3] == max(Material_Count):          # Prefer Aluminum instead of PM
        Final_Material_After_Change = 3
    elif Material_Count[4] == max(Material_Count):          # In case that PM elements are even to half of surrounding elements with majority
        Final_Material_After_Change = 4

print('Final_Material_After_Change', Final_Material_After_Change)




