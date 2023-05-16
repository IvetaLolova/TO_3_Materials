########################################################################################################################
########################################################################################################################
# Script for create geometry of line-start synchronous reluctance machine (3-material TO)
# 2023
# ----------------------------------------------
# this part of the project performs the following steps:
# -takes weighting coefficients w1/w2
# -creates normalized Gaussian networks (NGnet1/NGnet2)
# -creates the geometry of the rotor based on given conditions
# -checks the geometry feasibility from the point of view:
#     The iron part must be one connected piece
#     There can not be single elements in geometry.
#
# Ing. Iveta Lolová
########################################################################################################################
########################################################################################################################
# Import of libs
# ----------------------------------------------
import statistics
import os
import shutil
import numpy
import numpy as np
import math
from math import pi, cos, sin, sqrt
from random import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

########################################################################################################################
########################################################################################################################
# Setting directory
initial_directory = os.getcwd()
# print(initial_directory)
global Directory, Folder_Figures
Directory = initial_directory.replace("\\", "/")
print(Directory)

Folder_Figures = "Figures/"
if os.path.isdir(Folder_Figures):
    shutil.rmtree(Folder_Figures)
    os.mkdir(Folder_Figures)
elif not os.path.isdir(Folder_Figures):
    os.mkdir(Folder_Figures)
Folder_Figures = "/" + Folder_Figures + "/"
########################################################################################################################
########################################################################################################################
# GRAPHS setting
# Change font and size in graphs
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 17})

# Create the set of colors, that are used for final graphs.
colors = ["#ffffff", "#0000dd", "#cccccc", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF", "#FF00FF"]
# colors = [0=white, 1=dark_blue, 2=grey, 3=yellow, 4=red, 5=green, 6=light_blue, 7=purple]
my_cmap = ListedColormap(colors, name="My_ColorMap_TO")


########################################################################################################################
########################################################################################################################
# Defining of functions:


def plot_matrix(title_name, matrix, save=True):
    # Plot of final Geometry Matrix
    fig = plt.figure(figsize=(6.4, 3.5))
    fig.canvas.manager.set_window_title(title_name)
    plt.pcolormesh(matrix, vmin=0, vmax=len(colors), cmap=my_cmap, edgecolor='face', linewidth=0.25)
    plt.ylim(0, 35)
    plt.xlim(0, 90)
    plt.yticks([0, 5, 10, 15, 20, 25, 30, 35])
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    plt.xlabel('Element x-index (-)')
    plt.ylabel('Element y-index (-)')
    # cbar = plt.colorbar()
    # cbar.set_label("Colors Numbers (-)", labelpad=+1)

    # # Show the number or char in geometry matrix plot (if needed)
    # # For that the size of figure must be chanched (or size of font)
    # for (i, j), z in np.ndenumerate(Elements):
    #     plt.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')    #Shows numbers in Matrix z
    #     # plt.text(j, i, 'x', ha='center', va='center')                  #Shows char in 'char'

    plt.subplots_adjust(left=0.10, right=0.97, top=0.972, bottom=0.17)

    if save is True:
        Figure_Directory = Directory + Folder_Figures + title_name + ".pdf"
        fig.savefig(Figure_Directory, dpi=300)
        Figure_Directory = Directory + Folder_Figures + title_name + ".svg"
        fig.savefig(Figure_Directory, dpi=300)
    return 1


def create_NGnet(Name, sigma_fun, Overcross_fun, G_Diameter_in_fun, G_Diameter_out_fun, w_fun):
    print('\n-----------------------------------------')
    print('Calculation of NGnet', Name, ': Start!')
    G_Din = G_Diameter_in_fun  # Inner boundary condition ( Inner diameter for Gaussions function)
    G_Dout = G_Diameter_out_fun  # Outer boundary condition ( Outer diameter for Gaussions function)
    w = w_fun
    G_Rin = G_Din / 2  # Inner radius of Gauss
    G_Rout = G_Dout / 2  # Outer radius of Gauss

    Overcross = Overcross_fun  # překrytí gausů
    sigma = sigma_fun  # Variance
    Angle_of_symmetry = 45  # Axis of symmetry (90 pole is not symmetric and 45 pole is symmetric around x=y)

    sigma_x = sigma  # Variance x (CZ: Rozptyl x)
    sigma_y = sigma_x

    # A = ((sqrt(2 * np.pi) * sigma) ** (-1))  # Coefficient for normalizing Gaussian function
    A = 1  # In this optimalization the top of the Gauss is set to 1, because ON/OFF method

    # Creating meshgrid for Gaussians functions and also final for the final normalized Gaussian network
    Resolution = int((G_Rout * 10) + 1)  # Resolution of meshgrid
    x = np.linspace(0, G_Rout, num=Resolution)
    y = np.linspace(0, G_Rout, num=Resolution)
    x, y = np.meshgrid(x, y)

    # tic = time.perf_counter()
    # toc = time.perf_counter()
    # print(f"Time in {toc - tic:0.10f} seconds")

    Half_Computed = 2 * sqrt(2 * math.log(2)) * sigma * 4
    print("radius_computed:", Half_Computed)

    HalfRadiusGauss = Half_Computed

    b = []  # G_i/(SUM of all Gaussians function)
    f = 0  # f(x,y) = SUM(w_i * b_i(x,y))
    z = 0  # for showing results in plots

    # ----------------------------------------------
    # Calculation of centers of Gaussians functions

    G_Nrad = math.ceil(
        (G_Rout - G_Rin) / ((HalfRadiusGauss * 4 * (1 - Overcross)) / 10))  # Number of Gaussians function in radialy
    G_Nrad = round(G_Nrad)
    G_H = (G_Rout - G_Rin) / G_Nrad  # Height between two centers of Gaussians functions

    Gauss_XY = []  # xy - coordinate of gauss center
    Pocet_v_radku = []
    Number_of_Gauss = 0

    fig, ax = plt.subplots()
    TitleName = 'Contour_Suma_Gauss_' + Name
    fig.canvas.manager.set_window_title(TitleName)

    for i in range(G_Nrad):
        Gauss_XY.append([])

        r = G_Rin + 0.5 * G_H + i * G_H  # HalfRadiusGauss je půlka poloměru gausse

        perimetr = (2 * pi * r / (4))
        G_i = math.ceil(perimetr / ((HalfRadiusGauss * 4 * (90 / Angle_of_symmetry) * (1 - Overcross)) / 10))

        G_i = round(G_i)

        G_Angle_0 = Angle_of_symmetry / G_i  # Angle between two centers of Gaussians functions

        Pocet_v_radku.append([])
        Pocet_v_radku[i] = G_i

        # HalfRadiusGauss = HalfRadiusGauss*0.9

        for j in range(G_i):
            Gauss_XY[i].append([])
            Gauss_XY[i][j].append([])
            Gauss_XY[i][j].append([])

            angle = 0 + 0.5 * G_Angle_0 + j * G_Angle_0

            G_x = r * cos(angle * pi / 180)
            G_y = r * sin(angle * pi / 180)

            Gauss_XY[i][j][0] = G_x
            Gauss_XY[i][j][1] = G_y

            Number_of_Gauss = Number_of_Gauss + 1

        # HalfRadiusGauss = HalfRadiusGauss*0.9

    # Calculation of SUM of all Gaussians functions
    Suma_Gauss = 0
    for i in range(G_Nrad):
        for j in range(Pocet_v_radku[i]):
            my_x = Gauss_XY[i][j][0]
            my_y = Gauss_XY[i][j][1]

            Gauss_OneRound = A * np.exp(-((np.power(x - my_x, 2.) / (2 * np.power(sigma_x, 2.))) + (
                    np.power(y - my_y, 2.) / (2 * np.power(sigma_y, 2.)))))
            Suma_Gauss = Suma_Gauss + Gauss_OneRound

            z = Gauss_OneRound
            levels = [np.amax(z) / 5]
            CS = plt.contour(x, y, z, levels)
            # plt.Circle((5, 5), 0.5, color='b', fill=False)
        # sigma_x = sigma_x*0.9
        # sigma_y = sigma_y*0.9
    print("Number_of_Gauss:", Number_of_Gauss)

    # circle2 = plt.Circle((5, 5), 0.5, color='b', fill=False)
    # circle3 = plt.Circle((10, 10), 2, color='g', clip_on=False)
    #
    # plt.Circle((5, 5), 0.5, color='b', fill=False)
    # # fig.add_patch(circle2)
    # # ax.add_patch(circle3)

    ax.add_patch(plt.Circle((0, 0), (30 / 2), color='b', alpha=1, fill=False))
    ax.add_patch(plt.Circle((0, 0), (89.4 / 2), color='b', alpha=1, fill=False))

    plt.axis('square')
    plt.axis([0, 45, 0, 45])

    Figure_Directory = Directory + Folder_Figures + TitleName + ".pdf"
    fig.savefig(Figure_Directory, dpi=300)
    Figure_Directory = Directory + Folder_Figures + TitleName + ".svg"
    fig.savefig(Figure_Directory, dpi=300)

    # plt.show()

    # # Plot of the normalized Gaussian network
    # z = Suma_Gauss
    # fig = plt.figure()
    # fig.canvas.manager.set_window_title('Suma_Gauss________' + Name)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z, cmap=cm.jet)

    # plt.xlabel('x')
    # plt.ylabel('y')

    # plt.show()

    print('Calculation of Suma_Gauss: DONE!')

    # Calculation of Matrix b
    s = 0  # index of matrix of w
    for i in range(G_Nrad):
        b.append([])
        for j in range(Pocet_v_radku[i]):
            my_x = Gauss_XY[i][j][0]
            my_y = Gauss_XY[i][j][1]

            Gauss_OneRound = A * np.exp(-((np.power(x - my_x, 2.) / (2 * np.power(sigma_x, 2.))) + (
                    np.power(y - my_y, 2.) / (2 * np.power(sigma_y, 2.)))))

            b[i].append([])
            b[i][j] = Gauss_OneRound / Suma_Gauss

            f = f + w[s] * b[i][j]
            s = s + 1
            # sigma_x = sigma_x*0.9
            # sigma_y = sigma_y*0.9

    print('Calculation of b_i_j: DONE!')

    # Creating symmetric NGnet

    if Angle_of_symmetry == 45:

        f_sym = 0 * x + 0 * y

        for i in range(Resolution):
            for j in range(Resolution):
                f_Value = f[i, j]

                if j >= i:
                    f_sym[i, j] = f_Value

                if i <= j:
                    f_sym[j, i] = f_Value

        # z = f_sym
        #
        # fig = plt.figure()
        # fig.canvas.manager.set_window_title('NGnet'+Name)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, z, cmap=cm.jet)
        # # ax.set_title('NGnet')
        # # plt.xlabel('x')
        # # plt.ylabel('y')

        TitleName = 'NGnet_' + Name
        fig = plt.figure(figsize=(3.5, 3.5))
        fig.canvas.manager.set_window_title(TitleName)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, f_sym, cmap=cm.jet, vmax=1, vmin=0)
        plt.ylim(0, 45)
        plt.xlim(45, 0)
        # plt.clim(0,1)
        plt.yticks([0, 15, 30, 45])
        plt.xticks([0, 15, 30, 45])
        ax.set_zlim(0, 1)
        plt.xlabel('x (mm)')
        plt.ylabel('y(mm)')
        ax.view_init(46, -60)
        # ax.view_init(90, -90)
        plt.tight_layout()
        plt.subplots_adjust(top=1.0, bottom=0.105, left=0.0, right=0.892)

        Figure_Directory = Directory + Folder_Figures + TitleName + ".pdf"
        fig.savefig(Figure_Directory, dpi=300)
        Figure_Directory = Directory + Folder_Figures + TitleName + ".svg"
        fig.savefig(Figure_Directory, dpi=300)

        TitleName = 'NGnet_TOP_' + Name
        fig = plt.figure(figsize=(3.5, 3.5))
        fig.canvas.manager.set_window_title(TitleName)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, f_sym, cmap=cm.jet, vmax=1, vmin=0)
        # plt.axis('off')
        plt.ylim(0, 45)
        plt.xlim(0, 45)
        # plt.clim(0,1)
        plt.yticks([0, 15, 30, 45])
        plt.xticks([0, 15, 30, 45])
        ax.set_zticks([])
        # ax.set_zlim(emit=False)
        plt.xlabel('x (mm)')
        plt.ylabel('y(mm)')
        ax.view_init(89, -91)
        # ax.view_init(90, -90)
        plt.tight_layout()
        plt.subplots_adjust(top=1.2, bottom=-0.1, left=0.0, right=1.0)

        Figure_Directory = Directory + Folder_Figures + TitleName + ".pdf"
        fig.savefig(Figure_Directory, dpi=300)
        Figure_Directory = Directory + Folder_Figures + TitleName + ".svg"
        fig.savefig(Figure_Directory, dpi=300)

        # plt.show()

        # #Nice top view on NGnet (time-consuming + big size of the pdf and svg files
        # TitleName = 'NGnet_TOP' + Name
        # fig = plt.figure(figsize=(4.5, 3.5))
        # fig.canvas.manager.set_window_title(TitleName)
        # # plt.subplot()
        # plt.pcolor(x, y, f_sym, cmap=cm.jet, vmax=1, vmin=0)
        # plt.xlabel('x (mm)')
        # plt.ylabel('y (mm)')
        # cbar = plt.colorbar()
        # cbar.set_label("NGnet (-)", labelpad=+1)
        # plt.axis('square')
        # plt.axis((0, 45, 0, 45))
        # plt.yticks([0, 15, 30, 45])
        # plt.xticks([0, 15, 30, 45])
        # plt.subplots_adjust(top=0.965,bottom=0.18,left=0.145,right=0.945)
        # Figure_Directory = Directory + Folder_Figures + TitleName + ".pdf"
        # fig.savefig(Figure_Directory, dpi=300)
        # Figure_Directory = Directory + Folder_Figures + TitleName + ".svg"
        # fig.savefig(Figure_Directory, dpi=300)

        # plt.show()

    print('Calculation of NGnet', Name, ': DONE!\n')

    return f_sym


def material_test_single_element(material, elements_origin):
    print('Geometry test for single elements of one material!', str(material))
    Size = numpy.shape(elements_origin)
    N_rad_visited = Size[0]
    N_phr_visited = Size[1]
    # print('Velikost Elements', numpy.shape(elements_origin))
    elements_visited_material = []

    for i in range(N_rad_visited):
        elements_visited_material.append([])
        for j in range(N_phr_visited):
            elements_visited_material[i].append([])
            if elements_origin[i][j] == material:
                elements_visited_material[i][j] = 1
            else:
                elements_visited_material[i][j] = 0

    plot_matrix('04_Material_Visited_' + str(material), np.multiply(elements_visited_material, material))

    single_element = 0

    for i in range(N_rad_visited):
        for j in range(N_phr_visited):

            # Check for single element in every element except the edge elements
            if (elements_visited_material[i][j] == 1) and (0 < i < (N_rad_visited - 1) and 0 < j < (N_phr_visited - 1)):
                if (((elements_origin[i - 1][j] == material) or (elements_origin[i + 1][j] == material) or (
                        elements_origin[i][j - 1] == material) or (
                             elements_origin[i][j + 1] == material)) is not True):
                    single_element = single_element + 1
                    print('Single element exists!')
                    elements_visited_material[i][j] = 4
                else:
                    elements_visited_material[i][j] = 5

            # Check for single element in every element on the surface of rotor
            if ((elements_visited_material[i][j] == 1) and (
                    i == (N_rad_visited - 1) and 0 < j < (N_phr_visited - 1))):
                if (((elements_origin[i - 1][j] == material) or (elements_origin[i][j - 1] == material) or (
                        elements_origin[i][j + 1] == material)) is not True):
                    single_element = single_element + 1
                    print('Single element exists!')
                    elements_visited_material[i][j] = 4
                else:
                    elements_visited_material[i][j] = 5
    # If material is iron then set the inner line of elements that touch the shaft to iron.
    # Moreover, the edges of poles are set to iron.
    if material == 1:
        for i in range(N_rad_visited):
            for j in range(N_phr_visited):
                if i == 0 or j == 0 or j == (N_phr_visited - 1):
                    elements_visited_material[i][j] = 5

    if single_element > 0:
        print('Single elements exist! The number of them is:', single_element)
    else:
        print('Single element does not exist!')

    plot_matrix('04_Material_Visited_' + str(material) + "_Tested_single_element", elements_visited_material)

    return elements_visited_material, single_element


def change_single_element(material, elements_origin, elements_visited_material, single_element):
    print('Geometry test for single elements of one material!', str(material))
    Size = numpy.shape(elements_origin)
    N_rad_visited = Size[0]
    N_phr_visited = Size[1]

    for i in range(N_rad_visited):
        for j in range(N_phr_visited):
            if elements_visited_material == 4:
                # Changing element surrounded by four elements
                if 0 < i < (N_rad_visited - 1) and 0 < j < (N_phr_visited - 1):
                    Surrounding_materials = [elements_origin[i - 1][j], elements_origin[i + 1][j],
                                             elements_origin[i][j - 1], elements_origin[i][j + 1]]
                    statistics.mode(Surrounding_materials)

                    if (((elements_origin[i - 1][j] == material) or (elements_origin[i + 1][j] == material) or (
                            elements_origin[i][j - 1] == material) or (
                                 elements_origin[i][j + 1] == material)) is not True):
                        single_element = single_element + 1
                        print('Single element exists!')
                        elements_visited_material[i][j] = 4
                    else:
                        elements_visited_material[i][j] = 5

            # Check for single element in every element on the surface of rotor
            if (elements_visited_material[i][j] == 1) and (
                    i == (N_rad_visited - 1) and 0 < j < (N_phr_visited - 1)):
                if (((elements_origin[i - 1][j] == material) or (elements_origin[i][j - 1] == material) or (
                        elements_origin[i][j + 1] == material)) is not True):
                    single_element = single_element + 1
                    print('Single element exists!')
                    elements_visited_material[i][j] = 4
                else:
                    elements_visited_material[i][j] = 5
    # If material is iron then set the inner line of elements that touch the shaft to iron. Moreover, the edges of poles are set to iron.
    if material == 1:
        for i in range(N_rad_visited):
            for j in range(N_phr_visited):
                if i == 0 or j == 0 or j == (N_phr_visited - 1):
                    elements_visited_material[i][j] = 5

    if single_element > 0:
        print('Single elements exist! The number of them is:', single_element)
    else:
        print('Single element does not exist!')

    TitleName = 'material_Visited_' + str(material) + "_Tested_Sinle_Element"
    fig = plt.figure(figsize=(6.4, 3.5))
    fig.canvas.manager.set_window_title(TitleName)
    plt.pcolormesh(elements_visited_material, vmin=0, vmax=len(colors), cmap=my_cmap, edgecolor='face', linewidth=0.25)
    plt.ylim(0, 35)
    plt.xlabel('Element x-index (-)')
    plt.ylabel('Element y-index (-)')
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.94, top=0.985, bottom=0.19)

    Figure_Directory = Directory + Folder_Figures + TitleName + ".pdf"
    fig.savefig(Figure_Directory, dpi=300)
    Figure_Directory = Directory + Folder_Figures + TitleName + ".svg"
    fig.savefig(Figure_Directory, dpi=300)

    return elements_visited_material, single_element


def material_test_flying_part(material, elements_origin):
    print('Geometry test for flying elements of one material!', str(material))
    Size = numpy.shape(elements_origin)
    N_rad_visited = Size[0]
    N_phr_visited = Size[1]
    # print('Size of Elements:', numpy.shape(elements_origin))
    elements_visited_material = []
    color_green = 5
    color_purple = 7

    for i in range(N_rad_visited):
        elements_visited_material.append([])
        for j in range(N_phr_visited):
            elements_visited_material[i].append([])
            if elements_origin[i][j] == material:
                elements_visited_material[i][j] = 1
            else:
                elements_visited_material[i][j] = 0

    plot_matrix("Flying_Part_Visited_FUN_", np.multiply(elements_visited_material, material), save=True)

    elements_check_flying_part = elements_visited_material.copy()

    # If material is iron then set the inner line of elements that touch the shaft to iron.
    # Moreover, the edges of poles are set to iron.
    if material == 1:
        for i in range(N_rad_visited):
            for j in range(N_phr_visited):
                if i == 0 or j == 0 or j == (N_phr_visited - 1) or i == (N_rad_visited - 1):
                    elements_check_flying_part[i][j] = color_green

    change = 1
    while change == 1:
        change = 0
        for i in range(Nrad):
            for j in range(Nphr):

                # Up -> Down
                if (elements_visited_material[i][j] == 1) and i > 0 and j > 0 and i < (Nrad - 1) and j < (Nphr - 1):
                    if elements_check_flying_part[i - 1][j] == color_green or elements_check_flying_part[i + 1][
                        j] == color_green or elements_check_flying_part[i][j - 1] == color_green or \
                            elements_check_flying_part[i][j + 1] == color_green:
                        if elements_check_flying_part[i][j] != color_green:
                            elements_check_flying_part[i][j] = color_green
                            change = 1

                # Down -> Up
                k = Nrad - i - 1
                l = Nphr - j - 1

                if (elements_visited_material[k][l] == 1) and k > 0 and l > 0 and k < (Nrad - 1) and l < (Nphr - 1):
                    if elements_check_flying_part[k - 1][l] == color_green or elements_check_flying_part[k + 1][
                        l] == color_green or elements_check_flying_part[k][l - 1] == color_green or \
                            elements_check_flying_part[k][l + 1] == color_green:
                        if elements_check_flying_part[k][l] != color_green:
                            elements_check_flying_part[k][l] = color_green
                            change = 1
    flying_part = 0
    for i in range(N_rad_visited):
        for j in range(N_phr_visited):
            if elements_check_flying_part[i][j] == 1:
                elements_check_flying_part[i][j] = color_purple
                flying_part = flying_part + 1
    if flying_part > 0:
        print("Flying part exist in geometry!")
        print("Number of flying elements is:", flying_part)
    else:
        print("There is no flying part in geometry.")

    plot_matrix("Flying_Part_check", elements_check_flying_part)

    return elements_visited_material


########################################################################################################################
########################################################################################################################

# Parametres of rotor and number of block in radial and Number of peripheral blocks per pole
Din = 30  # SyMSpaceVar
Dout = 89.4  # SyMSpaceVar
Nrad = 35  # SyMSpaceVar
Nphr = 90  # SyMSpaceVar

Nrad = int(Nrad)
Nphr = int(Nphr)

Rin = Din / 2  # Inner radius of elements
Rout = Dout / 2  # Outer radius of elements
Height = (Rout - Rin) / Nrad  # Height of one block in created topology
Angle_0 = 90 / Nphr  # Angle of one block in created topology

########################################################################################################################
# Parameters of Gaussians function

G_Din = 20  # Inner boundary condition ( Inner diameter for Gaussions function)
G_Dout = 100  # Outer boundary condition ( Outer diameter for Gaussions function)
sigma = 2  # Variance  #SyMSpaceVar
sigma_x = sigma  # Variance x (CZ: Rozptyl x)
sigma_y = sigma_x

Overlap = 0.3  # překrytí gausů
########################################################################################################################
# Set up random weighting coefficients
w1 = [random() for i in range(500)]
w2 = [random() for i in range(500)]

# # Example of the defined weighting coefficients:
# w2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# w2 = [0.9650076067586537, 0.8042046501901402, 0.0076595368693391785, 0.6941627980217514, 0.8436057753210086, 0.4054061529994867, 0.9937326464738931, 0.20133914970329025, 0.6173074987764646, 0.6335738791739778, 0.4174151544521917, 0.8758056584381687, 0.6707264289774421, 0.24371814959926685, 0.7742276276949253, 0.8501934280003509, 0.020035543859127647, 0.11021774815570784, 0.5168296665211243, 0.5817858406296655, 0.20448275919567216, 0.4395476646513191, 0.04548964897381669, 0.42488168925490966, 0.04333748055525033, 0.32507984208963725, 0.31173686643716425, 0.22732228924631226, 0.6135236412696872, 0.8351919559287501, 0.6276580397104748, 0.37732485773256275, 0.49894750801774757, 0.5357115710276, 0.11458140825327245, 0.3828009432620748, 0.9370832676814479, 0.0935296549654826, 0.7503567757574535, 0.39421520662717924, 0.4734537969632904, 0.9637251619353142, 0.44348162753279474, 0.6080015583810475, 0.9456839459317692, 0.9007302809503839, 0.9642788234448715, 0.1324507151995692, 0.3612269860185122, 0.8300830728438218, 0.8125790681881178, 0.24252390292751635, 0.017076374359361157, 0.617969347260566, 0.9562431983508529, 0.6365749436512727, 0.7387714683966018, 0.5908619604811862, 0.37557437318253395, 0.5178175826548782, 0.393939249530879, 0.500413172380583, 0.713634677137668, 0.5237929449663083, 0.7588819272417107, 0.5443451114194632, 0.6640102395718604, 0.5882352233143361, 0.9462908948420264, 0.6686252842505969, 0.3590231216999946, 0.6611626627659034, 0.023583241251417086, 0.23422370914226254, 0.15149824338792872, 0.3730099849973799, 0.6139258495600275, 0.09763875392195687, 0.9547287812964103, 0.22456587561130747, 0.6140410172828111, 0.43508820136213733, 0.7798198396041081, 0.22854441303635342, 0.1685216531764281, 0.9956893058485291, 0.5795515669658081, 0.1298597141635226, 0.7194599552729518, 0.44337868194069907, 0.1312771479037268, 0.500799036842728, 0.19759323804335271, 0.8660532178726104, 0.04665629434890728, 0.39307616695581904, 0.4500909471000176, 0.44268016734612126, 0.8704253947318726, 0.795631651187295, 0.7394604985901222, 0.48948870815003176, 0.5161711290613483, 0.6625069501361694, 0.963977425981575, 0.1950806686267712, 0.7802121479050786, 0.4006924980001879, 0.8255198115706534, 0.4032628320182349, 0.06687404017440479, 0.683001536671522, 0.29606320962548904, 0.3649411563533649, 0.41084147446088515, 0.6835636024894114, 0.42172468508306504, 0.4302920837325891, 0.20125245689017524, 0.9204883180023378, 0.06730752043636978, 0.8815070928050751, 0.602886998593924, 0.6583992638912136, 0.4145796047529322, 0.012190952594760551, 0.23067551651487328, 0.34119740524876696, 0.5584556944885346, 0.5156854231056551, 0.5458835393485975, 0.8437338424546544, 0.6149945632750089, 0.5259273260346254, 0.6372908114980791, 0.7988210905389239, 0.8494495177562235, 0.04114349382099436, 0.5011163307348211, 0.6115542486445193, 0.4896791285687111, 0.4257053181215077, 0.7163172132798621, 0.7196550671811895, 0.697305807797201, 0.14201729526671902, 0.19517377244002898, 0.27877760621561287, 0.8105384357054081, 0.008680813398167508, 0.452196030916046, 0.24177837650786627, 0.8934571766304367, 0.08491919848409812, 0.362121577711764, 0.3902165324144782, 0.731502554205981, 0.23430553232712592, 0.4900906905193656, 0.38710892562819665, 0.13130663029728018, 0.3918916226002773, 0.6050204249096885, 0.9189629249066148, 0.7016570286885719, 0.9792295347846176, 0.1105138134514625, 0.7599126962088669, 0.8160120329310524, 0.6332017648478757, 0.7559396280175978, 0.18150775724375656, 0.2122161397192237, 0.5676900452933612, 0.5881787291112491, 0.18183011590509135, 0.4737589007328775, 0.11749864825242495, 0.8891701017385062, 0.5672020131621143, 0.36209755830039436, 0.24865941387778412, 0.03879943558172738, 0.7410356432240174, 0.893456967293674, 0.15287851281509424, 0.9456287801501737, 0.7813504858036527, 0.4406707346622114, 0.2935828001208425, 0.985888654896993, 0.9601907698003983, 0.6902476826158963, 0.20878425144435275, 0.8738041775164205, 0.5636589540614773, 0.95312749012739, 0.7211514785119324, 0.7799773239793214, 0.7160668626322266, 0.4921650252222689, 0.05681444556360449, 0.12159264921647162, 0.8225825320180479, 0.056141034008907464, 0.1274377212126523, 0.13266600133154693, 0.6064364348614307, 0.0880562791452526, 0.7846041397154844, 0.4636457398072279, 0.6314349826547335, 0.9836570230223948, 0.8489803718461736, 0.838113129542719, 0.20238960183725996, 0.2557990266040894, 0.37895466956117096, 0.4197370566932014, 0.319006767786269, 0.42443459879139267, 0.9146778481641239, 0.09339994451584421, 0.785935698214065, 0.056228994847702274, 0.46269342457072793, 0.35877233151852916, 0.3281636357273753, 0.5910121517609354, 0.28516206271544087, 0.34650257005840923, 0.8944218596461901, 0.7907607681849746, 0.4577145720991249, 0.9552191635435171, 0.23314707110753086, 0.10367757623118301, 0.6176282678922295, 0.6874917682950382, 0.9758175880821957, 0.2835398898660193, 0.42745886509233366, 0.9753187860471507, 0.28142913510581713, 0.3186056723934939, 0.06971065075172167, 0.714162999894654, 0.03848396665767151, 0.672729818148799, 0.8333157848451659, 0.3640660315676022, 0.9856823847227086, 0.9843966201549473, 0.12947005108382859, 0.9369595190498498, 0.8035101470944384, 0.16175178948364433, 0.9264160138004566, 0.19741331582214294, 0.262997264256576, 0.06476367043035192, 0.5748051721009086, 0.7988944757845887, 0.8087891455919292, 0.9172512172479851, 0.16257621997392013, 0.23484951152251743, 0.43080522985991665, 0.46509054062543387, 0.17588858668771867, 0.5950471732792355, 0.01072605078338762, 0.33652422868943943, 0.6827804779816178, 0.4723252997516336, 0.5380700392940441, 0.8044240621642141, 0.7730698210290589, 0.08691417442151661, 0.017187826256263805, 0.7426638524743068, 0.850996186714522, 0.5728112216657061, 0.7085669048168326, 0.012910808924855632, 0.33768434664809976, 0.351427354887495, 0.7208792139129704, 0.8758001367312698, 0.9666221282336914, 0.9608397898583217, 0.511956322780256, 0.4933015202558916, 0.7096488021548965, 0.055469652231998845, 0.452589401977894, 0.14072919691237273, 0.4751020537519043, 0.3565436320343849, 0.6536557230569603, 0.03507108198206488, 0.14795828258090227, 0.928803735269653, 0.37902850527713905, 0.018139266193371584, 0.595606189459138, 0.8540829881778803, 0.04574270205608044, 0.021067488789069277, 0.6787917303675531, 0.39893512398862196, 0.6659370019821036, 0.46096441840801383, 0.519619970010696, 0.2771120537838324, 0.7131214131982887, 0.19212151007694134, 0.5933335768739624, 0.11829808016992416, 0.5868350667803225, 0.8431458394045293, 0.3728977733546871, 0.6311407682033718, 0.3146418409574838, 0.362929363114045, 0.15194745584548508, 0.038717735615826365, 0.5059567529223843, 0.5938524990602723, 0.3153744693735382, 0.5007526192611798, 0.6869101787726123, 0.23273159860984516, 0.8664402480785955, 0.11987174066417094, 0.7419393804700521, 0.07571257991017177, 0.9539634941898597, 0.46089972396141443, 0.3066494448076931, 0.733934527039772, 0.6036332759509416, 0.7365182469712268, 0.47274921383648627, 0.052911517453136936, 0.9714220346961864, 0.604418753189099, 0.6476649839369408, 0.5693473070379469, 0.20718329289881043, 0.6886035891003828, 0.02392271059391704, 0.0914766764842253, 0.0026048413349981647, 0.7938803898712871, 0.5440146715159311, 0.6143167661904537, 0.9909273936644719, 0.09468707070910032, 0.9240843624790805, 0.4512121864614723, 0.08478916867314101, 0.010836429722165919, 0.5910683614303482, 0.49706342907185885, 0.20772787733746534, 0.3419580782308368, 0.31012470961616667, 0.4222299234091923, 0.1904116190792945, 0.21538676225529074, 0.9231239500011104, 0.14441893590162835, 0.43378550753665024, 0.5032486194630588, 0.5761019838557478, 0.4860243558572711, 0.8092262915467768, 0.658686751131634, 0.8084348580134568, 0.6367399212540734, 0.5855270828519362, 0.7447667518351904, 0.1205621867356953, 0.42090339518513986, 0.4804381800683234, 0.0448912906322716, 0.4724929150116898, 0.9743165210592927, 0.6290592351186266, 0.6012364890288464, 0.8209454245391888, 0.09062047050771449, 0.9628526533223303, 0.951452576889088, 0.4909215283576144, 0.45367264322912515, 0.0330154665527298, 0.6848013352197642, 0.45766083758971643, 0.5176556751219689, 0.1351702194800407, 0.4647746131024404, 0.7118152523416016, 0.01788597363411537, 0.9165964783484356, 0.03963371678152694, 0.514732498442911, 0.08666494921832224, 0.48702378172956107, 0.7530641663307723, 0.16923444375543806, 0.5209614044103202, 0.5443613146966565, 0.4246874746946431, 0.14005527168973864, 0.3768680791971093, 0.7845123354235396, 0.5474384698335362, 0.34404264841263155, 0.9645548071466599, 0.3030665202915024, 0.4856110351550462, 0.3870830195905913, 0.08529934641669867, 0.43996734424350525, 0.6059321181838483, 0.9478807707101651, 0.20534995990794547, 0.4156175562137814, 0.25186159193533375, 0.05675586887633055, 0.962935545322921, 0.4989436748269759, 0.16157714073501162, 0.8508252377908012, 0.23291207611520615, 0.9809831680699025, 0.12420163678200224, 0.2595674043257219, 0.5753680968356861, 0.5574845482392404, 0.22277909063639345, 0.8237164914605901, 0.3371237621673102, 0.5144262785650741, 0.39981056015631156, 0.23295583670336006, 0.1475051186967241, 0.35142952494488633, 0.15407348093633555, 0.9962135486216387, 0.019808711075909757, 0.8291700067560336, 0.8987877205782123, 0.008958134454488342, 0.3427890971323484, 0.6415084582110814, 0.17407780020178942, 0.795978345264957, 0.6659856134915103, 0.4875013274831781, 0.8227472798881886, 0.4721024749668672, 0.7673123830599756, 0.6266628523434465, 0.16264241794629286, 0.4361788652973274, 0.0836691365304768, 0.5815026204979191, 0.9292355000273478, 0.2206481181574893, 0.5568912931468981, 0.8397877255840939, 0.568260921099595, 0.11145136000259148, 0.9288341145618713, 0.7520107610807358, 0.7747070671876102, 0.5624132915253905, 0.19709420851251735, 0.010566682228486668, 0.6367518768693052, 0.9499870660672259, 0.4727767675071852, 0.08641598762950109, 0.9737374051537359, 0.35480784328359827, 0.32434967001687987, 0.11278834526462223, 0.5617014363769484, 0.616205715760868, 0.19706215865059984, 0.34435657584141566, 0.7498531867242192, 0.24551297198767597, 0.10685114717561195, 0.05673108103009916, 0.568537351914961, 0.5831271283225158]
# w1 = [0.7623808916181155, 0.2734293992597371, 0.5750505637046357, 0.13170496012931432, 0.8937926283215446, 0.6121276300952914, 0.9993308345514207, 0.3152336377808015, 0.41728428464023337, 0.317814629310734, 0.8500891679905107, 0.5969402597877763, 0.04506556933592576, 0.9666731417346152, 0.4556484941800305, 0.925746415124271, 0.5506048057092856, 0.9033517801178309, 0.7525514204805708, 0.22935719226387763, 0.8950028646432512, 0.28276730184138466, 0.9834197505171983, 0.09944851810017452, 0.6409601336341068, 0.8581937537061461, 0.3488336286771584, 0.7961708335042914, 0.6081190137367983, 0.4948339067752475, 0.886247063413163, 0.12179824030021913, 0.4820073157227489, 0.2407989111258566, 0.047737920245804144, 0.8342446876356918, 0.6527503332973935, 0.06872951104177971, 0.9336771948912669, 0.10310547946161996, 0.14068742079389196, 0.5226973600778133, 0.35398252132311847, 0.5288960633711964, 0.9866222674867389, 0.11756185762326465, 0.47827867885001063, 0.5251562800848899, 0.5776436340284606, 0.4829526556199659, 0.5364896874219233, 0.9586311953317355, 0.017333725520214727, 0.08663298607975056, 0.470580391221608, 0.9839509163995814, 0.29161509509816297, 0.8136484196836913, 0.7722607826902125, 0.7955432157246904, 0.6883688340804558, 0.6453830471993923, 0.25942953802760105, 0.9145820265926269, 0.9724298853985139, 0.8112690334118882, 0.18726076168308292, 0.3187013138373175, 0.07376126662705318, 0.6337686020879573, 0.4303559070487314, 0.9863968798930314, 0.18354322599967754, 0.7502167316604264, 0.24022640481901314, 0.10809243673357938, 0.17577359501688994, 0.1363019913613538, 0.5405035952347294, 0.9722694092360975, 0.8999684674568156, 0.44137742577276995, 0.41759101218231787, 0.9343126188521442, 0.5189529098633595, 0.5008390914069091, 0.586459906055137, 0.514593603325584, 0.45226147849208387, 0.3753747243327039, 0.10932203745961966, 0.8412525615245924, 0.5637646198912885, 0.6532715504191756, 0.7767086072579958, 0.11695375772799654, 0.34255342543288925, 0.846998526482136, 0.7266527311916898, 0.6860303104529153, 0.11963497460437467, 0.5264225352628705, 0.8749003344373122, 0.2884692133584015, 0.4674921580636979, 0.20198110238447486, 0.8383386389735051, 0.24799838602749313, 0.47848995895103663, 0.6638238880006885, 0.6475082852175009, 0.555237480268709, 0.4342314049376813, 0.8871200652131248, 0.27589303697953294, 0.4426861334267992, 0.23069453792597716, 0.2547188427619834, 0.42686422526924395, 0.8230450385601544, 0.5137068608851977, 0.7023538203399657, 0.42633135402234457, 0.7900295986486698, 0.20671267791170544, 0.2846847701884534, 0.858182997436183, 0.4442673168302783, 0.9275889985831218, 0.12241157960474836, 0.10020578339054975, 0.3063366305700991, 0.2526630339039101, 0.7781272393010659, 0.7585914610749996, 0.9479155142151154, 0.4325738203864121, 0.2690364765647322, 0.8430450501146389, 0.056266285839536745, 0.1756979114083682, 0.9372120421456889, 0.7968011866465018, 0.33706300377082377, 0.8143619107443886, 0.9711897803222829, 0.1291930048552803, 0.4140707558472948, 0.03606939093088557, 0.9925762337397699, 0.42658819424973793, 0.9262092983520843, 0.2948331852001782, 0.42367377122058913, 0.5509737553072732, 0.8141393115383742, 0.6190995405887689, 0.7219178768762916, 0.08014050603128675, 0.5427579297594336, 0.3726372195628771, 0.33388742403481697, 0.6268101566865043, 0.6461105278400269, 0.0816532917433812, 0.3428977321840687, 0.5537148148191533, 0.5488505231170641, 0.08442099444958417, 0.027753965642922762, 0.5988099171763676, 0.06504009591788895, 0.14211762116124416, 0.5100818138510784, 0.4892785281890043, 0.2794000384696471, 0.7507474367336271, 0.974332615282347, 0.7241843353406683, 0.3129407920699063, 0.28608895464462336, 0.20553694785050947, 0.6191293344538239, 0.25795188165474436, 0.0707326713499783, 0.9200650109586128, 0.4753718179803794, 0.39698244244249314, 0.181501652978712, 0.6043976977330968, 0.4588945335890301, 0.19088179755472656, 0.9653083211991953, 0.21376489478469907, 0.21835833694720752, 0.5002449564328032, 0.24906279045312163, 0.5964238490557043, 0.9717686882599355, 0.5857571747481289, 0.30212443933917177, 0.6418205427866058, 0.9535281761572376, 0.06815607506086596, 0.5611428468842156, 0.7479606516838964, 0.11299903668748057, 0.1487797941906256, 0.5981572466702554, 0.7314969412505236, 0.07383860409404142, 0.9108355835267943, 0.3758667885662724, 0.984638151668239, 0.31120890731150985, 0.31501137741381047, 0.3177486732181287, 0.9507378143009148, 0.12484969466447504, 0.3172257005536061, 0.8716855317984712, 0.7090652813402338, 0.6573266124588829, 0.005370777376295766, 0.6295776689512544, 0.04874893013174253, 0.6765178032266586, 0.1964573464713819, 0.13755893953226428, 0.10283837685060482, 0.6379470843176342, 0.1175650687918608, 0.7693415780845924, 0.32053860753375496, 0.23270436034156217, 0.03731896437842974, 0.5056621250804592, 0.09254987389114522, 0.9369052456091914, 0.339132113731462, 0.030075423286152647, 0.8987151674658249, 0.7000458156978336, 0.5628256325678772, 0.022117437005063523, 0.4063760565951039, 0.2688506460626975, 0.4637738049186, 0.8264459383788233, 0.6068904644692449, 0.9571881210378024, 0.6922687981638583, 0.9023425491716639, 0.0552881778105806, 0.3242604776915854, 0.49671142077481467, 0.6195123789868192, 0.9186865143543881, 0.08489469769438018, 0.9681477607152095, 0.36754328287908156, 0.9922607715445889, 0.21375004661638863, 0.3519604169562245, 0.6464800959353779, 0.6408252668090317, 0.4721782255468675, 0.6895113198623833, 0.4954418740772667, 0.6345560417561468, 0.5190999908007294, 0.3271748339090472, 0.21411906581270135, 0.6526251713989153, 0.06985995394136679, 0.9544697070677571, 0.6675114364051963, 0.6342670272413511, 0.6464269777084466, 0.15131369870196232, 0.22919301935802494, 0.3183046389846179, 0.6714826278668364, 0.12465196427890757, 0.5207689143441363, 0.04712382701824469, 0.5590509203594439, 0.07547623791447311, 0.8520304724292955, 0.3993378946580002, 0.8048679009392664, 0.43272014748764376, 0.7908706134194553, 0.6424396864359394, 0.6115432240958314, 0.9246233216172781, 0.6553233350567168, 0.49908391998102386, 0.3659923824721252, 0.38183067442987595, 0.20183388813088998, 0.13472368129753953, 0.2264545851235379, 0.14854602498061553, 0.8236879564089349, 0.4938199656234432, 0.34642372094468665, 0.03126820245481943, 0.4693127197868243, 0.9777697354577419, 0.9298974039071662, 0.7236447598199238, 0.21644772723301597, 0.4819003446066359, 0.19085299558875812, 0.37557424807784245, 0.5095418317359106, 0.31992318696368205, 0.22640546229544734, 0.3210087663840069, 0.03589820795623344, 0.5259684166400974, 0.6761959056849844, 0.5477314147329425, 0.6274040104922008, 0.7239082326824865, 0.7677080169206619, 0.8767662657242782, 0.5340351259022028, 0.1173464210273979, 0.5750679175475077, 0.6783546013765458, 0.4202627018403008, 0.21716232191407547, 0.6814374259332368, 0.30258395804466576, 0.2799238880146502, 0.034243866583645866, 0.9452213906418977, 0.82576274309846, 0.8082156364583015, 0.6370630846768883, 0.7733929327873393, 0.43277686739952315, 0.866477235409209, 0.2696980010180242, 0.8489143006610619, 0.5494714212922168, 0.08607449305258441, 0.636162339519783, 0.8360678478292143, 0.4205359555856809, 0.6889651387181647, 0.6325955997419745, 0.9963421606335287, 0.6744019458293367, 0.5353143586011565, 0.5899934213114996, 0.6407598221027575, 0.7374655758033417, 0.6380678836350057, 0.692154824435332, 0.9533261420051745, 0.4800737942544736, 0.7163011878708392, 0.12988423343244004, 0.27164613719583763, 0.17862601111748966, 0.7545461502305203, 0.08217217946652378, 0.07811096388938443, 0.5420829836564349, 0.5600978952214604, 0.30747762908843723, 0.6330007868372197, 0.629057466178237, 0.11614967783208163, 0.09677288278078322, 0.42593420483685707, 0.39333269651986813, 0.8252810116681902, 0.5445585580043741, 0.03546620726219318, 0.44300018780162487, 0.46282538063773293, 0.968887209918573, 0.9340453452493056, 0.08490125752655875, 0.10697817187176129, 0.2465854619370642, 0.4179995287091264, 0.21757217300852438, 0.8730929587357352, 0.12709361007136333, 0.1217083669298874, 0.07681286848087077, 0.7376030460494332, 0.6358535003559329, 0.6552336980431924, 0.809666128405567, 0.5403405336725163, 0.9620916393452609, 0.41255518796572244, 0.7405106359987836, 0.20870909203686572, 0.9246741673050304, 0.3655485317625845, 0.5771207040440263, 0.6286549485446462, 0.28201223178585433, 0.7369559316107702, 0.026867831081109683, 0.6561219407491222, 0.40316911509776165, 0.07392877832502287, 0.16959088022411006, 0.5468163492723179, 0.7585014562499999, 0.7199394783973219, 0.8069382451031515, 0.16560265960606502, 0.02086815663883157, 0.35808751089360613, 0.32755651927601737, 0.41023309774229455, 0.48377126562101436, 0.0038902975982122445, 0.8601336207446081, 0.10812700166290634, 0.7847369612919503, 0.6714572354558438, 0.35475450668986663, 0.23518149867826477, 0.44234876494505326, 0.8546505216585327, 0.6557269441827348, 0.44434441806256, 0.2249532245401984, 0.7687794201798231, 0.5258472263335755, 0.03647459390108587, 0.7036490977666066, 0.01961613908298876, 0.7764318771662609, 0.5775788672964065, 0.7898500297117204, 0.041812618972240756, 0.9423949202044606, 0.2142372263241561, 0.24155252998020382, 0.10284281528568584, 0.7358495338285788, 0.8909010136869799, 0.5985207730798049, 0.08245732367800451, 0.5216862054861248, 0.6152596163464732, 0.4191542208444422, 0.6869175047514164, 0.07037476622996475, 0.6512203613953443, 0.6246959799810717, 0.7815095439618008, 0.6388372514130114, 0.4076107295443845, 0.9187810895521888, 0.9359989030451776, 0.8505755151261933, 0.7189467685227026, 0.08859969346337093, 0.3055427890962614, 0.42319901633945156, 0.7451767160471717, 0.13403502893986008, 0.5223501737314638, 0.09218745387181515, 0.31498526495102674, 0.8487774930249044, 0.21670505335948453, 0.26962340593185274, 0.967646848276871, 0.4218934379867306, 0.14142338972779922, 0.36321491115696813, 0.7241976502177032, 0.6029742141887281, 0.7213275012779828, 0.8095762201444757, 0.7836770411947517, 0.8333405257368842, 0.43792818097425235, 0.8140844774706366, 0.1037794079391835, 0.5736100546042298, 0.6066683513580898, 0.8936432526693425, 0.9008685097515425, 0.2588634627226014, 0.36247037112250013, 0.47597709326334336]

########################################################################################################################
# Calling function NGnet to create two NGnets
# There are different variants for quick check of final NGnets:
# # NGnet1:
# NGnet1 = create_NGnet("1", sigma, Overlap, G_Din, G_Dout,w1)
# NGnet1 = create_NGnet("1", 2, 0.3, 20, 100, w1)
NGnet1 = create_NGnet("1", 1.2, 0.2, 40, 90, w1)
# NGnet1 = create_NGnet("1", 1.1, 0.25, 38, 90, w1)  # Number_of_Gauss: 78
# # NGnet2:
NGnet2 = create_NGnet("2", 2.5, +0.1, 50, 90, w2)
# NGnet2 = create_NGnet("2", 1, +0.2, 20, 90,w2)

# Now there are two diffrent NGnets defined. In next steps the final geometry will be defined.
########################################################################################################################
# Defining final geometry in form of matrix.
# The matrix of elements is then used for defining materials of each element in Ansys Maxwell software.

# Variables for Matrix evaluating each Element in created topology
Elements = []
# Elements_flying_part = []
# Elements_show = []

n = 0
stringIron = ''
stringVacuum = ''
ArrayVacuum = []
stringPM = ''
stringAluminum = ''

material_Iron = 1
material_Air = 2
material_Aluminium = 3
material_PM = 4

# Creation of blank matrix of elements with given size
for i in range(Nrad):
    Elements.append([])
    for j in range(Nphr):
        Elements[i].append([])

for i in range(Nrad):
    for j in range(Nphr):
        # Calculation value of f(x,y) in centers of each elements
        r = Rin + 0.5 * Height + i * Height
        angle = 0 + 0.5 * Angle_0 + j * Angle_0
        Value_x = r * cos(angle * pi / 180) * 10
        Value_y = r * sin(angle * pi / 180) * 10

        Value_x = round(Value_x)
        Value_y = round(Value_y)

        # Creation of matrix, which copies the values of NGnets within givin number of elements
        f_NGnet1 = NGnet1[Value_x, Value_y]
        f_NGnet2 = NGnet2[Value_x, Value_y]

        # Defining Geometry based on conditions for two NGnet:
        if (f_NGnet1 > 0.5) or i == 0 or j == 0 or j == (Nphr - 1):
            Elements[i][j] = material_Iron
            stringIron = stringIron + 'Pixel_' + str(i) + '_' + str(j) + ','
        elif f_NGnet1 <= 0.5 and (f_NGnet2 > 0.5):
            Elements[i][j] = material_Air
            stringVacuum = stringVacuum + 'Pixel_' + str(i) + '_' + str(j) + ','
            string = 'Pixel_' + str(i) + '_' + str(j)
            ArrayVacuum.append([])
            ArrayVacuum[n] = string
            n += 1
        elif f_NGnet1 <= 0.5 and f_NGnet2 <= 0.5:
            Elements[i][j] = material_Aluminium
            stringVacuum = stringVacuum + 'Pixel_' + str(i) + '_' + str(j) + ','
            string = 'Pixel_' + str(i) + '_' + str(j)

plot_matrix('01_Geometry_MATRIX', Elements)
print('The final Geometry Matrix was created.')
# The final Geometry Matrix was created.
########################################################################################################################
# FEASIBILITY OF GEOMETRY

# Now, the geometry will be checked if there are no single if elements.
# Note: Single element is considered as an element that is not connected
# to another element with the same material.
print('Geometry check starts! - Single element check!')
# Checking all materials separately:
material_test_single_element(material_Iron, Elements)
material_test_single_element(material_Air, Elements)
material_test_single_element(material_Aluminium, Elements)
# material_test_single_element(material_PM, Elements)

# Now, the geometry will be checked if iron is one piece.
print('Geometry check starts! - Flying parts check')
material_test_flying_part(material_Iron, Elements)

########################################################################################################################
print('END of script!')
########################################################################################################################
########################################################################################################################
########################################################################################################################
