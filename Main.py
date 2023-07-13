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
from datetime import datetime

########################################################################################################################
########################################################################################################################
# Setting directory
initial_directory = os.getcwd()
# print(initial_directory)
global Directory, Folder_Figures
Directory = initial_directory.replace("\\", "/")
print(Directory)

Folder_Figures = "Plots/"
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

print('The final Geometry Matrix was created.', datetime.now())


########################################################################################################################
########################################################################################################################
# Defining of functions for NGnet and creation of final GEOMETRY:


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
        # Figure_Directory = Directory + Folder_Figures + title_name + ".svg"
        # fig.savefig(Figure_Directory, dpi=300)
    return 1


def create_NGnet(name, sigma_fun, overcross_fun, gauss_diameter_inner_fun, gauss_diameter_outer_fun, w_fun):
    # This function: create_NGnet() - creates the NGnet (the normalized Gaussian network)
    # Explanation of parameters:
    # name                      - name of created NGnet. Currently the numbers are used.
    # signma_fun                - sigma is variance of Gaussian function (it defines width of Gaussian function)
    # overcross_fun             - define overcross of two Gaussian functions deployed in optimized space
    # gauss_diameter_inner_fun  - Inner boundary condition ( Inner diameter for Gaussions function)
    # gauss_diameter_outer_fun  - Outer boundary condition ( Outer diameter for Gaussions function)
    print('-----------------------------------------')
    print('Calculation of NGnet', name, ': Start!')

    gauss_radius_inner = gauss_diameter_inner_fun / 2  # Inner radius of Gauss
    gauss_radius_outer = gauss_diameter_outer_fun / 2  # Outer radius of Gauss

    Angle_of_symmetry = 45  # Axis of symmetry (90 pole is not symmetric and 45 pole is symmetric around x=y)

    sigma_x = sigma_fun  # Variance x (CZ: Rozptyl x)
    sigma_y = sigma_x

    # A = ((sqrt(2 * np.pi) * sigma) ** (-1))  # Coefficient for normalizing Gaussian function
    A = 1  # In this optimalization the top of the Gauss is set to 1, because ON/OFF method

    # Creating meshgrid for Gaussians functions and also final for the final normalized Gaussian network
    Resolution = int((gauss_radius_outer * 10) + 1)  # Resolution of meshgrid
    x = np.linspace(0, gauss_radius_outer, num=Resolution)
    y = np.linspace(0, gauss_radius_outer, num=Resolution)
    x, y = np.meshgrid(x, y)

    # tic = time.perf_counter()
    # toc = time.perf_counter()
    # print(f"Time in {toc - tic:0.10f} seconds")

    Half_Computed = 2 * sqrt(2 * math.log(2)) * sigma_fun * 4
    print("radius_computed:", Half_Computed)

    HalfRadiusGauss = Half_Computed

    b = []  # G_i/(SUM of all Gaussians function)
    f = 0  # f(x,y) = SUM(w_i * b_i(x,y))
    z = 0  # for showing results in plots

    # ----------------------------------------------
    # Calculation of centers of Gaussians functions

    G_Nrad = math.ceil(
        (gauss_radius_outer - gauss_radius_inner) / ((HalfRadiusGauss * 4 * (1 - overcross_fun)) / 10))  # Number of Gaussians function in radialy
    G_Nrad = round(G_Nrad)
    G_H = (gauss_radius_outer - gauss_radius_inner) / G_Nrad  # Height between two centers of Gaussians functions

    Gauss_XY = []  # xy - coordinate of gauss center
    Pocet_v_radku = []
    Number_of_Gauss = 0

    fig, ax = plt.subplots()
    TitleName = 'Contour_Suma_Gauss_' + name
    fig.canvas.manager.set_window_title(TitleName)

    for i in range(G_Nrad):
        Gauss_XY.append([])

        half_radius_gauss_fun = gauss_radius_inner + 0.5 * G_H + i * G_H  # HalfRadiusGauss je půlka poloměru gausse

        perimeter = (2 * pi * half_radius_gauss_fun / 4)
        G_i = math.ceil(perimeter / ((HalfRadiusGauss * 4 * (90 / Angle_of_symmetry) * (1 - overcross_fun)) / 10))

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

            G_x = half_radius_gauss_fun * cos(angle * pi / 180)
            G_y = half_radius_gauss_fun * sin(angle * pi / 180)

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

            f = f + w_fun[s] * b[i][j]
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

        TitleName = 'NGnet_' + name
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

        TitleName = 'NGnet_TOP_' + name
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

        # #Plot - Nice top view on NGnet (time-consuming + big size of the pdf and svg files
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

    print('Calculation of NGnet', name, ': DONE!\n')

    return f_sym


def material_test_single_element(material, elements_origin):
    print('Geometry test for single elements of one material!', str(material))
    Size = numpy.shape(elements_origin)
    N_rad_visited = Size[0]
    N_phr_visited = Size[1]
    # print('Velikost Elements', numpy.shape(elements_origin))

    color_red = 4
    color_green = 5

    elements_visited_material = []

    for i in range(N_rad_visited):
        elements_visited_material.append([])
        for j in range(N_phr_visited):
            elements_visited_material[i].append([])
            if elements_origin[i][j] == material:
                elements_visited_material[i][j] = 1
            else:
                elements_visited_material[i][j] = 0

    plot_matrix('04_'+ str(material) +'_Material_Visited', np.multiply(elements_visited_material, material))

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
                    elements_visited_material[i][j] = color_red
                else:
                    elements_visited_material[i][j] = color_green

            # Check for single element in every element on the surface of rotor
            if ((elements_visited_material[i][j] == 1) and (
                    i == (N_rad_visited - 1) and 0 < j < (N_phr_visited - 1))):
                if (((elements_origin[i - 1][j] == material) or (elements_origin[i][j - 1] == material) or (
                        elements_origin[i][j + 1] == material)) is not True):
                    single_element = single_element + 1
                    print('Single element exists!')
                    elements_visited_material[i][j] = color_red
                else:
                    elements_visited_material[i][j] = color_green
    # If material is iron then set the inner line of elements that touch the shaft to iron.
    # Moreover, the edges of poles are set to iron.
    if material == 1:
        for i in range(N_rad_visited):
            for j in range(N_phr_visited):
                if i == 0 or j == 0 or j == (N_phr_visited - 1):
                    elements_visited_material[i][j] = color_green

    plot_matrix('04_'+ str(material) +'_Material_Visited_Tested_single_element', elements_visited_material)

    if single_element > 0:
        print('Single elements exist! The number of them is:', single_element)
        plot_matrix('04_'+ str(material) + '_elements_origin_before_change', elements_origin)
        change_single_element(material,elements_origin,elements_visited_material)
        plot_matrix('04_'+ str(material) + '_elements_origin_after_change', elements_origin)
    else:
        print('Single element does not exist!')

    return elements_visited_material, single_element


def change_single_element(material, elements_origin, elements_visited_material):
    print('Geometry change of single elements of one material! Material:', str(material))
    Size = numpy.shape(elements_origin)
    N_rad_visited = Size[0]
    N_phr_visited = Size[1]

    count_change_single_element = 0

    color_red = 4
    color_green = 5

    surrounding_materials = []

    for i in range(N_rad_visited):
        for j in range(N_phr_visited):
            if elements_visited_material[i][j] == color_red:
                if 0 < i:
                    surrounding_materials.append(elements_origin[i - 1][j])
                if i < (N_rad_visited - 1):
                    surrounding_materials.append(elements_origin[i + 1][j])
                if 0 < j:
                    surrounding_materials.append(elements_origin[i][j - 1])
                if j < (N_phr_visited - 1):
                    surrounding_materials.append(elements_origin[i][j + 1])

                # Material_Iron      = 1
                # Material_Air       = 2
                # Material_Aluminium = 3
                # Material_PM        = 4

                material_count = [0, 0, 0, 0, 0]  # Materials_Count = [None, Iron, Air, Aluminium, PM]
                for k in range(len(material_count)):
                    material_count[k] = surrounding_materials.count(k)

                final_material_after_change = 0

                if max(material_count) > (len(surrounding_materials) / 2):
                    # if maximal count of surrounding materials is bigger then half of surrounding elements,
                    # than change single_element material to that material of majority of surrounding elements
                    final_material_after_change = material_count.index(max(material_count))
                elif max(material_count) <= (len(surrounding_materials) / 2):
                    # if maximal count of surrounding materials is lower or even to half of surrounding elements,
                    # than change single_element material to that material as follows starting with Air as prefered material:
                    if material_count[2] == max(material_count):  # Prefer Air instead of Iron/Aluminum/PM
                        final_material_after_change = 2
                    elif material_count[1] == max(material_count):  # Prefer Iron instead of Aluminum/PM
                        final_material_after_change = 1
                    elif material_count[3] == max(material_count):  # Prefer Aluminum instead of PM
                        final_material_after_change = 3
                    elif material_count[4] == max(
                            material_count):  # In case that PM elements are even to half of surrounding elements with majority
                        final_material_after_change = 4

                elements_origin[i][j] = final_material_after_change

                print('Element[', i,'][', j,'] was changed from material:', material,' to material:', final_material_after_change,'.')

                count_change_single_element = count_change_single_element + 1

    print('element_origin was succsefully changed: ', count_change_single_element, '-times.')
    return count_change_single_element


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
# Set up random weighting coefficients
w1 = [random() for i in range(500)]
w2 = [random() for i in range(500)]

print(w1)
print(w2)
# # Example of the defined weighting coefficients:
# w2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# w2 = [0.9650076067586537, 0.8042046501901402, 0.0076595368693391785, 0.6941627980217514, 0.8436057753210086, 0.4054061529994867, 0.9937326464738931, 0.20133914970329025, 0.6173074987764646, 0.6335738791739778, 0.4174151544521917, 0.8758056584381687, 0.6707264289774421, 0.24371814959926685, 0.7742276276949253, 0.8501934280003509, 0.020035543859127647, 0.11021774815570784, 0.5168296665211243, 0.5817858406296655, 0.20448275919567216, 0.4395476646513191, 0.04548964897381669, 0.42488168925490966, 0.04333748055525033, 0.32507984208963725, 0.31173686643716425, 0.22732228924631226, 0.6135236412696872, 0.8351919559287501, 0.6276580397104748, 0.37732485773256275, 0.49894750801774757, 0.5357115710276, 0.11458140825327245, 0.3828009432620748, 0.9370832676814479, 0.0935296549654826, 0.7503567757574535, 0.39421520662717924, 0.4734537969632904, 0.9637251619353142, 0.44348162753279474, 0.6080015583810475, 0.9456839459317692, 0.9007302809503839, 0.9642788234448715, 0.1324507151995692, 0.3612269860185122, 0.8300830728438218, 0.8125790681881178, 0.24252390292751635, 0.017076374359361157, 0.617969347260566, 0.9562431983508529, 0.6365749436512727, 0.7387714683966018, 0.5908619604811862, 0.37557437318253395, 0.5178175826548782, 0.393939249530879, 0.500413172380583, 0.713634677137668, 0.5237929449663083, 0.7588819272417107, 0.5443451114194632, 0.6640102395718604, 0.5882352233143361, 0.9462908948420264, 0.6686252842505969, 0.3590231216999946, 0.6611626627659034, 0.023583241251417086, 0.23422370914226254, 0.15149824338792872, 0.3730099849973799, 0.6139258495600275, 0.09763875392195687, 0.9547287812964103, 0.22456587561130747, 0.6140410172828111, 0.43508820136213733, 0.7798198396041081, 0.22854441303635342, 0.1685216531764281, 0.9956893058485291, 0.5795515669658081, 0.1298597141635226, 0.7194599552729518, 0.44337868194069907, 0.1312771479037268, 0.500799036842728, 0.19759323804335271, 0.8660532178726104, 0.04665629434890728, 0.39307616695581904, 0.4500909471000176, 0.44268016734612126, 0.8704253947318726, 0.795631651187295, 0.7394604985901222, 0.48948870815003176, 0.5161711290613483, 0.6625069501361694, 0.963977425981575, 0.1950806686267712, 0.7802121479050786, 0.4006924980001879, 0.8255198115706534, 0.4032628320182349, 0.06687404017440479, 0.683001536671522, 0.29606320962548904, 0.3649411563533649, 0.41084147446088515, 0.6835636024894114, 0.42172468508306504, 0.4302920837325891, 0.20125245689017524, 0.9204883180023378, 0.06730752043636978, 0.8815070928050751, 0.602886998593924, 0.6583992638912136, 0.4145796047529322, 0.012190952594760551, 0.23067551651487328, 0.34119740524876696, 0.5584556944885346, 0.5156854231056551, 0.5458835393485975, 0.8437338424546544, 0.6149945632750089, 0.5259273260346254, 0.6372908114980791, 0.7988210905389239, 0.8494495177562235, 0.04114349382099436, 0.5011163307348211, 0.6115542486445193, 0.4896791285687111, 0.4257053181215077, 0.7163172132798621, 0.7196550671811895, 0.697305807797201, 0.14201729526671902, 0.19517377244002898, 0.27877760621561287, 0.8105384357054081, 0.008680813398167508, 0.452196030916046, 0.24177837650786627, 0.8934571766304367, 0.08491919848409812, 0.362121577711764, 0.3902165324144782, 0.731502554205981, 0.23430553232712592, 0.4900906905193656, 0.38710892562819665, 0.13130663029728018, 0.3918916226002773, 0.6050204249096885, 0.9189629249066148, 0.7016570286885719, 0.9792295347846176, 0.1105138134514625, 0.7599126962088669, 0.8160120329310524, 0.6332017648478757, 0.7559396280175978, 0.18150775724375656, 0.2122161397192237, 0.5676900452933612, 0.5881787291112491, 0.18183011590509135, 0.4737589007328775, 0.11749864825242495, 0.8891701017385062, 0.5672020131621143, 0.36209755830039436, 0.24865941387778412, 0.03879943558172738, 0.7410356432240174, 0.893456967293674, 0.15287851281509424, 0.9456287801501737, 0.7813504858036527, 0.4406707346622114, 0.2935828001208425, 0.985888654896993, 0.9601907698003983, 0.6902476826158963, 0.20878425144435275, 0.8738041775164205, 0.5636589540614773, 0.95312749012739, 0.7211514785119324, 0.7799773239793214, 0.7160668626322266, 0.4921650252222689, 0.05681444556360449, 0.12159264921647162, 0.8225825320180479, 0.056141034008907464, 0.1274377212126523, 0.13266600133154693, 0.6064364348614307, 0.0880562791452526, 0.7846041397154844, 0.4636457398072279, 0.6314349826547335, 0.9836570230223948, 0.8489803718461736, 0.838113129542719, 0.20238960183725996, 0.2557990266040894, 0.37895466956117096, 0.4197370566932014, 0.319006767786269, 0.42443459879139267, 0.9146778481641239, 0.09339994451584421, 0.785935698214065, 0.056228994847702274, 0.46269342457072793, 0.35877233151852916, 0.3281636357273753, 0.5910121517609354, 0.28516206271544087, 0.34650257005840923, 0.8944218596461901, 0.7907607681849746, 0.4577145720991249, 0.9552191635435171, 0.23314707110753086, 0.10367757623118301, 0.6176282678922295, 0.6874917682950382, 0.9758175880821957, 0.2835398898660193, 0.42745886509233366, 0.9753187860471507, 0.28142913510581713, 0.3186056723934939, 0.06971065075172167, 0.714162999894654, 0.03848396665767151, 0.672729818148799, 0.8333157848451659, 0.3640660315676022, 0.9856823847227086, 0.9843966201549473, 0.12947005108382859, 0.9369595190498498, 0.8035101470944384, 0.16175178948364433, 0.9264160138004566, 0.19741331582214294, 0.262997264256576, 0.06476367043035192, 0.5748051721009086, 0.7988944757845887, 0.8087891455919292, 0.9172512172479851, 0.16257621997392013, 0.23484951152251743, 0.43080522985991665, 0.46509054062543387, 0.17588858668771867, 0.5950471732792355, 0.01072605078338762, 0.33652422868943943, 0.6827804779816178, 0.4723252997516336, 0.5380700392940441, 0.8044240621642141, 0.7730698210290589, 0.08691417442151661, 0.017187826256263805, 0.7426638524743068, 0.850996186714522, 0.5728112216657061, 0.7085669048168326, 0.012910808924855632, 0.33768434664809976, 0.351427354887495, 0.7208792139129704, 0.8758001367312698, 0.9666221282336914, 0.9608397898583217, 0.511956322780256, 0.4933015202558916, 0.7096488021548965, 0.055469652231998845, 0.452589401977894, 0.14072919691237273, 0.4751020537519043, 0.3565436320343849, 0.6536557230569603, 0.03507108198206488, 0.14795828258090227, 0.928803735269653, 0.37902850527713905, 0.018139266193371584, 0.595606189459138, 0.8540829881778803, 0.04574270205608044, 0.021067488789069277, 0.6787917303675531, 0.39893512398862196, 0.6659370019821036, 0.46096441840801383, 0.519619970010696, 0.2771120537838324, 0.7131214131982887, 0.19212151007694134, 0.5933335768739624, 0.11829808016992416, 0.5868350667803225, 0.8431458394045293, 0.3728977733546871, 0.6311407682033718, 0.3146418409574838, 0.362929363114045, 0.15194745584548508, 0.038717735615826365, 0.5059567529223843, 0.5938524990602723, 0.3153744693735382, 0.5007526192611798, 0.6869101787726123, 0.23273159860984516, 0.8664402480785955, 0.11987174066417094, 0.7419393804700521, 0.07571257991017177, 0.9539634941898597, 0.46089972396141443, 0.3066494448076931, 0.733934527039772, 0.6036332759509416, 0.7365182469712268, 0.47274921383648627, 0.052911517453136936, 0.9714220346961864, 0.604418753189099, 0.6476649839369408, 0.5693473070379469, 0.20718329289881043, 0.6886035891003828, 0.02392271059391704, 0.0914766764842253, 0.0026048413349981647, 0.7938803898712871, 0.5440146715159311, 0.6143167661904537, 0.9909273936644719, 0.09468707070910032, 0.9240843624790805, 0.4512121864614723, 0.08478916867314101, 0.010836429722165919, 0.5910683614303482, 0.49706342907185885, 0.20772787733746534, 0.3419580782308368, 0.31012470961616667, 0.4222299234091923, 0.1904116190792945, 0.21538676225529074, 0.9231239500011104, 0.14441893590162835, 0.43378550753665024, 0.5032486194630588, 0.5761019838557478, 0.4860243558572711, 0.8092262915467768, 0.658686751131634, 0.8084348580134568, 0.6367399212540734, 0.5855270828519362, 0.7447667518351904, 0.1205621867356953, 0.42090339518513986, 0.4804381800683234, 0.0448912906322716, 0.4724929150116898, 0.9743165210592927, 0.6290592351186266, 0.6012364890288464, 0.8209454245391888, 0.09062047050771449, 0.9628526533223303, 0.951452576889088, 0.4909215283576144, 0.45367264322912515, 0.0330154665527298, 0.6848013352197642, 0.45766083758971643, 0.5176556751219689, 0.1351702194800407, 0.4647746131024404, 0.7118152523416016, 0.01788597363411537, 0.9165964783484356, 0.03963371678152694, 0.514732498442911, 0.08666494921832224, 0.48702378172956107, 0.7530641663307723, 0.16923444375543806, 0.5209614044103202, 0.5443613146966565, 0.4246874746946431, 0.14005527168973864, 0.3768680791971093, 0.7845123354235396, 0.5474384698335362, 0.34404264841263155, 0.9645548071466599, 0.3030665202915024, 0.4856110351550462, 0.3870830195905913, 0.08529934641669867, 0.43996734424350525, 0.6059321181838483, 0.9478807707101651, 0.20534995990794547, 0.4156175562137814, 0.25186159193533375, 0.05675586887633055, 0.962935545322921, 0.4989436748269759, 0.16157714073501162, 0.8508252377908012, 0.23291207611520615, 0.9809831680699025, 0.12420163678200224, 0.2595674043257219, 0.5753680968356861, 0.5574845482392404, 0.22277909063639345, 0.8237164914605901, 0.3371237621673102, 0.5144262785650741, 0.39981056015631156, 0.23295583670336006, 0.1475051186967241, 0.35142952494488633, 0.15407348093633555, 0.9962135486216387, 0.019808711075909757, 0.8291700067560336, 0.8987877205782123, 0.008958134454488342, 0.3427890971323484, 0.6415084582110814, 0.17407780020178942, 0.795978345264957, 0.6659856134915103, 0.4875013274831781, 0.8227472798881886, 0.4721024749668672, 0.7673123830599756, 0.6266628523434465, 0.16264241794629286, 0.4361788652973274, 0.0836691365304768, 0.5815026204979191, 0.9292355000273478, 0.2206481181574893, 0.5568912931468981, 0.8397877255840939, 0.568260921099595, 0.11145136000259148, 0.9288341145618713, 0.7520107610807358, 0.7747070671876102, 0.5624132915253905, 0.19709420851251735, 0.010566682228486668, 0.6367518768693052, 0.9499870660672259, 0.4727767675071852, 0.08641598762950109, 0.9737374051537359, 0.35480784328359827, 0.32434967001687987, 0.11278834526462223, 0.5617014363769484, 0.616205715760868, 0.19706215865059984, 0.34435657584141566, 0.7498531867242192, 0.24551297198767597, 0.10685114717561195, 0.05673108103009916, 0.568537351914961, 0.5831271283225158]
# w1 = [0.7623808916181155, 0.2734293992597371, 0.5750505637046357, 0.13170496012931432, 0.8937926283215446, 0.6121276300952914, 0.9993308345514207, 0.3152336377808015, 0.41728428464023337, 0.317814629310734, 0.8500891679905107, 0.5969402597877763, 0.04506556933592576, 0.9666731417346152, 0.4556484941800305, 0.925746415124271, 0.5506048057092856, 0.9033517801178309, 0.7525514204805708, 0.22935719226387763, 0.8950028646432512, 0.28276730184138466, 0.9834197505171983, 0.09944851810017452, 0.6409601336341068, 0.8581937537061461, 0.3488336286771584, 0.7961708335042914, 0.6081190137367983, 0.4948339067752475, 0.886247063413163, 0.12179824030021913, 0.4820073157227489, 0.2407989111258566, 0.047737920245804144, 0.8342446876356918, 0.6527503332973935, 0.06872951104177971, 0.9336771948912669, 0.10310547946161996, 0.14068742079389196, 0.5226973600778133, 0.35398252132311847, 0.5288960633711964, 0.9866222674867389, 0.11756185762326465, 0.47827867885001063, 0.5251562800848899, 0.5776436340284606, 0.4829526556199659, 0.5364896874219233, 0.9586311953317355, 0.017333725520214727, 0.08663298607975056, 0.470580391221608, 0.9839509163995814, 0.29161509509816297, 0.8136484196836913, 0.7722607826902125, 0.7955432157246904, 0.6883688340804558, 0.6453830471993923, 0.25942953802760105, 0.9145820265926269, 0.9724298853985139, 0.8112690334118882, 0.18726076168308292, 0.3187013138373175, 0.07376126662705318, 0.6337686020879573, 0.4303559070487314, 0.9863968798930314, 0.18354322599967754, 0.7502167316604264, 0.24022640481901314, 0.10809243673357938, 0.17577359501688994, 0.1363019913613538, 0.5405035952347294, 0.9722694092360975, 0.8999684674568156, 0.44137742577276995, 0.41759101218231787, 0.9343126188521442, 0.5189529098633595, 0.5008390914069091, 0.586459906055137, 0.514593603325584, 0.45226147849208387, 0.3753747243327039, 0.10932203745961966, 0.8412525615245924, 0.5637646198912885, 0.6532715504191756, 0.7767086072579958, 0.11695375772799654, 0.34255342543288925, 0.846998526482136, 0.7266527311916898, 0.6860303104529153, 0.11963497460437467, 0.5264225352628705, 0.8749003344373122, 0.2884692133584015, 0.4674921580636979, 0.20198110238447486, 0.8383386389735051, 0.24799838602749313, 0.47848995895103663, 0.6638238880006885, 0.6475082852175009, 0.555237480268709, 0.4342314049376813, 0.8871200652131248, 0.27589303697953294, 0.4426861334267992, 0.23069453792597716, 0.2547188427619834, 0.42686422526924395, 0.8230450385601544, 0.5137068608851977, 0.7023538203399657, 0.42633135402234457, 0.7900295986486698, 0.20671267791170544, 0.2846847701884534, 0.858182997436183, 0.4442673168302783, 0.9275889985831218, 0.12241157960474836, 0.10020578339054975, 0.3063366305700991, 0.2526630339039101, 0.7781272393010659, 0.7585914610749996, 0.9479155142151154, 0.4325738203864121, 0.2690364765647322, 0.8430450501146389, 0.056266285839536745, 0.1756979114083682, 0.9372120421456889, 0.7968011866465018, 0.33706300377082377, 0.8143619107443886, 0.9711897803222829, 0.1291930048552803, 0.4140707558472948, 0.03606939093088557, 0.9925762337397699, 0.42658819424973793, 0.9262092983520843, 0.2948331852001782, 0.42367377122058913, 0.5509737553072732, 0.8141393115383742, 0.6190995405887689, 0.7219178768762916, 0.08014050603128675, 0.5427579297594336, 0.3726372195628771, 0.33388742403481697, 0.6268101566865043, 0.6461105278400269, 0.0816532917433812, 0.3428977321840687, 0.5537148148191533, 0.5488505231170641, 0.08442099444958417, 0.027753965642922762, 0.5988099171763676, 0.06504009591788895, 0.14211762116124416, 0.5100818138510784, 0.4892785281890043, 0.2794000384696471, 0.7507474367336271, 0.974332615282347, 0.7241843353406683, 0.3129407920699063, 0.28608895464462336, 0.20553694785050947, 0.6191293344538239, 0.25795188165474436, 0.0707326713499783, 0.9200650109586128, 0.4753718179803794, 0.39698244244249314, 0.181501652978712, 0.6043976977330968, 0.4588945335890301, 0.19088179755472656, 0.9653083211991953, 0.21376489478469907, 0.21835833694720752, 0.5002449564328032, 0.24906279045312163, 0.5964238490557043, 0.9717686882599355, 0.5857571747481289, 0.30212443933917177, 0.6418205427866058, 0.9535281761572376, 0.06815607506086596, 0.5611428468842156, 0.7479606516838964, 0.11299903668748057, 0.1487797941906256, 0.5981572466702554, 0.7314969412505236, 0.07383860409404142, 0.9108355835267943, 0.3758667885662724, 0.984638151668239, 0.31120890731150985, 0.31501137741381047, 0.3177486732181287, 0.9507378143009148, 0.12484969466447504, 0.3172257005536061, 0.8716855317984712, 0.7090652813402338, 0.6573266124588829, 0.005370777376295766, 0.6295776689512544, 0.04874893013174253, 0.6765178032266586, 0.1964573464713819, 0.13755893953226428, 0.10283837685060482, 0.6379470843176342, 0.1175650687918608, 0.7693415780845924, 0.32053860753375496, 0.23270436034156217, 0.03731896437842974, 0.5056621250804592, 0.09254987389114522, 0.9369052456091914, 0.339132113731462, 0.030075423286152647, 0.8987151674658249, 0.7000458156978336, 0.5628256325678772, 0.022117437005063523, 0.4063760565951039, 0.2688506460626975, 0.4637738049186, 0.8264459383788233, 0.6068904644692449, 0.9571881210378024, 0.6922687981638583, 0.9023425491716639, 0.0552881778105806, 0.3242604776915854, 0.49671142077481467, 0.6195123789868192, 0.9186865143543881, 0.08489469769438018, 0.9681477607152095, 0.36754328287908156, 0.9922607715445889, 0.21375004661638863, 0.3519604169562245, 0.6464800959353779, 0.6408252668090317, 0.4721782255468675, 0.6895113198623833, 0.4954418740772667, 0.6345560417561468, 0.5190999908007294, 0.3271748339090472, 0.21411906581270135, 0.6526251713989153, 0.06985995394136679, 0.9544697070677571, 0.6675114364051963, 0.6342670272413511, 0.6464269777084466, 0.15131369870196232, 0.22919301935802494, 0.3183046389846179, 0.6714826278668364, 0.12465196427890757, 0.5207689143441363, 0.04712382701824469, 0.5590509203594439, 0.07547623791447311, 0.8520304724292955, 0.3993378946580002, 0.8048679009392664, 0.43272014748764376, 0.7908706134194553, 0.6424396864359394, 0.6115432240958314, 0.9246233216172781, 0.6553233350567168, 0.49908391998102386, 0.3659923824721252, 0.38183067442987595, 0.20183388813088998, 0.13472368129753953, 0.2264545851235379, 0.14854602498061553, 0.8236879564089349, 0.4938199656234432, 0.34642372094468665, 0.03126820245481943, 0.4693127197868243, 0.9777697354577419, 0.9298974039071662, 0.7236447598199238, 0.21644772723301597, 0.4819003446066359, 0.19085299558875812, 0.37557424807784245, 0.5095418317359106, 0.31992318696368205, 0.22640546229544734, 0.3210087663840069, 0.03589820795623344, 0.5259684166400974, 0.6761959056849844, 0.5477314147329425, 0.6274040104922008, 0.7239082326824865, 0.7677080169206619, 0.8767662657242782, 0.5340351259022028, 0.1173464210273979, 0.5750679175475077, 0.6783546013765458, 0.4202627018403008, 0.21716232191407547, 0.6814374259332368, 0.30258395804466576, 0.2799238880146502, 0.034243866583645866, 0.9452213906418977, 0.82576274309846, 0.8082156364583015, 0.6370630846768883, 0.7733929327873393, 0.43277686739952315, 0.866477235409209, 0.2696980010180242, 0.8489143006610619, 0.5494714212922168, 0.08607449305258441, 0.636162339519783, 0.8360678478292143, 0.4205359555856809, 0.6889651387181647, 0.6325955997419745, 0.9963421606335287, 0.6744019458293367, 0.5353143586011565, 0.5899934213114996, 0.6407598221027575, 0.7374655758033417, 0.6380678836350057, 0.692154824435332, 0.9533261420051745, 0.4800737942544736, 0.7163011878708392, 0.12988423343244004, 0.27164613719583763, 0.17862601111748966, 0.7545461502305203, 0.08217217946652378, 0.07811096388938443, 0.5420829836564349, 0.5600978952214604, 0.30747762908843723, 0.6330007868372197, 0.629057466178237, 0.11614967783208163, 0.09677288278078322, 0.42593420483685707, 0.39333269651986813, 0.8252810116681902, 0.5445585580043741, 0.03546620726219318, 0.44300018780162487, 0.46282538063773293, 0.968887209918573, 0.9340453452493056, 0.08490125752655875, 0.10697817187176129, 0.2465854619370642, 0.4179995287091264, 0.21757217300852438, 0.8730929587357352, 0.12709361007136333, 0.1217083669298874, 0.07681286848087077, 0.7376030460494332, 0.6358535003559329, 0.6552336980431924, 0.809666128405567, 0.5403405336725163, 0.9620916393452609, 0.41255518796572244, 0.7405106359987836, 0.20870909203686572, 0.9246741673050304, 0.3655485317625845, 0.5771207040440263, 0.6286549485446462, 0.28201223178585433, 0.7369559316107702, 0.026867831081109683, 0.6561219407491222, 0.40316911509776165, 0.07392877832502287, 0.16959088022411006, 0.5468163492723179, 0.7585014562499999, 0.7199394783973219, 0.8069382451031515, 0.16560265960606502, 0.02086815663883157, 0.35808751089360613, 0.32755651927601737, 0.41023309774229455, 0.48377126562101436, 0.0038902975982122445, 0.8601336207446081, 0.10812700166290634, 0.7847369612919503, 0.6714572354558438, 0.35475450668986663, 0.23518149867826477, 0.44234876494505326, 0.8546505216585327, 0.6557269441827348, 0.44434441806256, 0.2249532245401984, 0.7687794201798231, 0.5258472263335755, 0.03647459390108587, 0.7036490977666066, 0.01961613908298876, 0.7764318771662609, 0.5775788672964065, 0.7898500297117204, 0.041812618972240756, 0.9423949202044606, 0.2142372263241561, 0.24155252998020382, 0.10284281528568584, 0.7358495338285788, 0.8909010136869799, 0.5985207730798049, 0.08245732367800451, 0.5216862054861248, 0.6152596163464732, 0.4191542208444422, 0.6869175047514164, 0.07037476622996475, 0.6512203613953443, 0.6246959799810717, 0.7815095439618008, 0.6388372514130114, 0.4076107295443845, 0.9187810895521888, 0.9359989030451776, 0.8505755151261933, 0.7189467685227026, 0.08859969346337093, 0.3055427890962614, 0.42319901633945156, 0.7451767160471717, 0.13403502893986008, 0.5223501737314638, 0.09218745387181515, 0.31498526495102674, 0.8487774930249044, 0.21670505335948453, 0.26962340593185274, 0.967646848276871, 0.4218934379867306, 0.14142338972779922, 0.36321491115696813, 0.7241976502177032, 0.6029742141887281, 0.7213275012779828, 0.8095762201444757, 0.7836770411947517, 0.8333405257368842, 0.43792818097425235, 0.8140844774706366, 0.1037794079391835, 0.5736100546042298, 0.6066683513580898, 0.8936432526693425, 0.9008685097515425, 0.2588634627226014, 0.36247037112250013, 0.47597709326334336]
w1 = [0.3034193503492044, 0.24007680890071403, 0.31602949079480913, 0.6300612961277586, 0.9227669795425613, 0.9308835546508797, 0.4675116268406144, 0.46387568884894204, 0.8786245557850598, 0.35015549959415615, 0.3377221164945118, 0.8806402303760511, 0.7301943444345738, 0.6250229150106805, 0.624453450885767, 0.7402265951893059, 0.9407429758796181, 0.4363746600552916, 0.6950205675210699, 0.8551649372051872, 0.886372014558808, 0.27503579345284823, 0.12090481776788398, 0.894776574890421, 0.38566497017595347, 0.4784996168209381, 0.3807181737220894, 0.8035581497049035, 0.09924062078847218, 0.704420626721175, 0.331328503958343, 0.9817515361912142, 0.26031107982735235, 0.9229492087879273, 0.6275818459721839, 0.14149119788913977, 0.6092306640321987, 0.9088687903535326, 0.7125973992657821, 0.4425101713687817, 0.756488408375943, 0.7008604137076175, 0.6562432254900383, 0.8343241868413493, 0.5505950824091427, 0.662783293355372, 0.3232714282750484, 0.8568475022071361, 0.1216977383062936, 0.7009531089610093, 0.4390665611105474, 0.766801082528488, 0.4369340649917617, 0.3577304769452455, 0.4692559734296039, 0.8214696121847648, 0.9997519620823645, 0.564168714383173, 0.7143538374392276, 0.9505247912730586, 0.04570666503978804, 0.5051201115782867, 0.5962995858259172, 0.8235057082894542, 0.12733947272367718, 0.7387774300970128, 0.6235831315286684, 0.88559534851906, 0.7143134925233671, 0.3970464872411795, 0.7816868631354094, 0.55772687525804, 0.08193632098053005, 0.4001129170435789, 0.42578125288443536, 0.9267576206314111, 0.9841850447407896, 0.9712542455313252, 0.36146317871860667, 0.1601522204174316, 0.913930811869491, 0.425440879620654, 0.8828020256185678, 0.2849413485395137, 0.26761838981196073, 0.3862872400464272, 0.17746444896919789, 0.29712092113219135, 0.7153157448857648, 0.22855965697722624, 0.08709738205958861, 0.8828350387168524, 0.47004739759086134, 0.2150735517008907, 0.0878057978497313, 0.9428600377239057, 0.2895967810844755, 0.3517420480652612, 0.8210104762355066, 0.554075557094171, 0.019096625787263233, 0.9652667334033036, 0.7926404569497514, 0.11905441804976147, 0.6739392020712934, 0.3155113170765502, 0.10691764740366383, 0.6184604926235839, 0.19125676855889318, 0.9161084513347091, 0.1684882141648142, 0.13881741200703657, 0.6188747399031007, 0.41415170688658387, 0.9797997776516353, 0.6369544454201018, 0.6921872122232959, 0.7376751493247345, 0.5607667082100609, 0.06731499515367123, 0.6803853511175507, 0.9095948696604487, 0.46397383734974373, 0.377507114141189, 0.15656347787284508, 0.4903062700506534, 0.011406355299308668, 0.4890540830061798, 0.4470834453986987, 0.319437680817523, 0.37420636795135154, 0.8123861534180806, 0.13188406968466038, 0.6060797126945555, 0.33895277159641735, 0.7868576754039894, 0.5763861834732943, 0.891141617997187, 0.7086598962326395, 0.4300130894735935, 0.10740325065721101, 0.8681031606107442, 0.1179311405400929, 0.720242070059484, 0.100438178374604, 0.25064222264747005, 0.7771516246164559, 0.8567081934834423, 0.8033536548214639, 0.6592958783457721, 0.4251558050853492, 0.5597287227453868, 0.05093189650156438, 0.5345212712633706, 0.8155722753410771, 0.9796393728130891, 0.24100897908767915, 0.3169154412834302, 0.52795041039933, 0.42475965462222176, 0.544471490413067, 0.5286074699849526, 0.45906855385543255, 0.8828007044203403, 0.9231407055508273, 0.6511509845001173, 0.1540348756782709, 0.6599105261212974, 0.8246285692544651, 0.6222798801089449, 0.43291541955378354, 0.11176609695647455, 0.10369789850115041, 0.49168923605225123, 0.9350814015181864, 0.4651617965883489, 0.18521733518640382, 0.8991109671976277, 0.4252904870958698, 0.4227122998461793, 0.4710768453819535, 0.8573301280439815, 0.4171462703993055, 0.4876659119009876, 0.5854486419128673, 0.5720071174550884, 0.5231345621528476, 0.24116583063297914, 0.5466972902900147, 0.7219144484514056, 0.7486859897047915, 0.9546199955672484, 0.5011375057774605, 0.966038760002348, 0.25639022276328916, 0.2226695501977184, 0.8342517908174681, 0.27554275684019147, 0.3346393124989939, 0.8071886014412196, 0.8482468143206163, 0.04991052147561226, 0.08502763042736639, 0.5261593557450084, 0.4369323334565801, 0.9513123101003813, 0.42957103354730475, 0.5021650907742855, 0.30669466039035875, 0.8770083064735908, 0.41272121041453513, 0.0744028606866407, 0.7545077281580352, 0.6296233054634968, 0.14859542297774075, 0.755671296430315, 0.2280510505490626, 0.4526615387292776, 0.60992818938585, 0.5965886181036423, 0.9512623670705338, 0.9496106268029593, 0.32224081312178265, 0.5069456245095115, 0.577380682962512, 0.5570931117390516, 0.800280470433793, 0.41242827676983085, 0.1144582185244083, 0.48444361208746534, 0.008891376901031367, 0.9196045631537535, 0.7290036250144717, 0.08586170070016386, 0.3716268184366025, 0.9246785153936368, 0.48147251692273585, 0.5157821383227009, 0.4857715506581086, 0.6004268409478546, 0.13804868537959325, 0.5934719990090693, 0.5605610995075164, 0.3599303604784976, 0.42918276963038426, 0.7453989854753159, 0.4219644874497851, 0.23643750021127707, 0.19430963197369, 0.11964450718398567, 0.20872572892694052, 0.4170709962780178, 0.7102909943596468, 0.04153708498848541, 0.29990399096541176, 0.9142722305974939, 0.22026040831940252, 0.6531565733898069, 0.7800928016921852, 0.6563736588385832, 0.3631552179415781, 0.18748501858860744, 0.6798338428472849, 0.7979507245778455, 0.32691487924048646, 0.9467804467195015, 0.2058486480568098, 0.815081399963895, 0.17419844668061102, 0.4984954338579838, 0.060947672157593114, 0.6494662122198495, 0.032260982484985656, 0.426840601096092, 0.6335245760895359, 0.7154536383235831, 0.18681616923634148, 0.275080157453109, 0.7965021304159353, 0.13353903410553936, 0.6688111142820544, 0.9308199966821084, 0.03908324881132341, 0.05123257630704936, 0.8825103992588843, 0.13996689708265742, 0.6405135753990232, 0.25073265683894197, 0.6285902199427216, 0.7412559644613825, 0.7329562516348497, 0.20405780130591356, 0.10853635286519303, 0.42595267146849647, 0.4065073475630637, 0.057072181813747846, 0.9915135452545736, 0.49390370041729414, 0.44592479495063975, 0.1910522135522028, 0.9745697735954417, 0.9837112301956485, 0.04593744843364389, 0.641642096725723, 0.7207600654008102, 0.18553267228311165, 0.7147997591727967, 0.5409293135663852, 0.9443331065705775, 0.6611180840722799, 0.31908929441959855, 0.7508155183939423, 0.32293868755793476, 0.81966571067523, 0.2829649582505589, 0.9851500414801072, 0.09985811490766427, 0.7082834713409739, 0.028103720486221673, 0.3862786645654267, 0.23081889376690812, 0.12087101135986, 0.2363911654488382, 0.45616539926380006, 0.9910214487381397, 0.3312337597792757, 0.030259929796515372, 0.8392471429642766, 0.5867036300015233, 0.8496718806119546, 0.43945033066181005, 0.14724897503070622, 0.10671514227248169, 0.600155523846116, 0.6725976775165181, 0.8532954323901658, 0.6090452443947887, 0.5878849481388494, 0.7073770947322338, 0.35463238808785646, 0.5976905099843571, 0.4007107605470015, 0.8337888285423316, 0.6446949009770365, 0.7325092647831242, 0.4347623507146925, 0.6510080938754714, 0.9096655686261801, 0.5003216147554838, 0.5619166052274353, 0.4867207138971047, 0.7388024087077089, 0.2487948846575716, 0.6632785005337539, 0.26965662340036867, 0.8304388910003657, 0.013009639797289196, 0.6204791474750561, 0.343732482154675, 0.27676304245162797, 0.8782739088713624, 0.4016275406780997, 0.20451622030937977, 0.6255744689771648, 0.3323802052028084, 0.674682012799484, 0.578957407533214, 0.6661881203771006, 0.9543935980439959, 0.4841102958520561, 0.4040192787639164, 0.10600869045159655, 0.8585397351939023, 0.9023202526968659, 0.06137277187328305, 0.21368135449775416, 0.5547407520147707, 0.09736982568377073, 0.4300770551520723, 0.2479520239011357, 0.6364492664938732, 0.3804863831887154, 0.3012542245350781, 0.3794199828565298, 0.9447265620713037, 0.32869328675801257, 0.36974334705927436, 0.8322545180433221, 0.7021352272344171, 0.15710093419677262, 0.3650465401049857, 0.6249661588144726, 0.7057147592736807, 0.574865302597981, 0.6367389565618813, 0.9324749892053991, 0.721493925999726, 0.5678989827749831, 0.2848205611702226, 0.35266328204356134, 0.58797404375563, 0.8324299381673309, 0.16693903735658755, 0.4088346473627388, 0.44205722568889694, 0.8533839977453284, 0.9612634854019636, 0.5989135566131583, 0.01891855216800653, 0.8749671819489276, 0.9325886296266347, 0.648016420902679, 0.41282050510684287, 0.8591182867811936, 0.3058929129316117, 0.2787094717908858, 0.7701200414626947, 0.8205597591634248, 0.523405747420455, 0.5119497685377482, 0.8140798877662776, 0.5659978287367732, 0.11162664240278575, 0.3610422865471442, 0.3229729989620471, 0.9646119287524709, 0.1195303833005471, 0.4516020784116377, 0.20406736881000598, 0.9296432749816962, 0.2550516626866882, 0.4579367607742868, 0.626945628229262, 0.42236453157790066, 0.785022629722063, 0.8853061962337019, 0.19519887546501602, 0.6731282481990012, 0.2731677521736836, 0.9548986886255304, 0.3503862782882271, 0.8690858011543497, 0.6552114498065837, 0.978326977978551, 0.41671332378882275, 0.7322606152337725, 0.23152934983594997, 0.5499768694088738, 0.7544737050074681, 0.4920029380351302, 0.7957362380872823, 0.44477538885330525, 0.6182293932927587, 0.26626232211277445, 0.20780849389576805, 0.03947806091221673, 0.29794994119406415, 0.8823147266855673, 0.683421669332105, 0.983436918627967, 0.5926699280043638, 0.26912999060222853, 0.058293521301288176, 0.90099820810411, 0.3664527897320614, 0.14615364825335941, 0.5936249865243441, 0.9721612506863695, 0.9466260204254971, 0.14129154867465343, 0.5255119056893442, 0.10713645851129916, 0.8524449411232078, 0.5150613225822613, 0.7342120095780169, 0.402604257914442, 0.3674700933067998, 0.9422918935001812, 0.5100650279693553, 0.5372522712154197, 0.3074703537140837, 0.7444721118387256, 0.7494826250020882, 0.07301751227975817, 0.3017354506863973, 0.3567575922341203, 0.6832377523568538, 0.737295772421348, 0.5971271594671068, 0.7903687613239798, 0.1385983651602417, 0.3292534337063251, 0.9387809958054917, 0.2763208744057879, 0.7272860599181176, 0.3414051730175345, 0.5836993510257665, 0.8237102814851263, 0.807269453811553, 0.6116977698044955]
w2 = [0.7926191714602882, 0.5014751243367791, 0.8987369440380633, 0.8292249198779176, 0.5517863665671111, 0.18966077109780566, 0.564914654673355, 0.09091496553151757, 0.994817972713339, 0.44993691967215965, 0.6887835701150995, 0.7912312900333172, 0.9400535746379459, 0.7848195913052388, 0.9150674927208229, 0.8654530077142832, 0.8300673593506374, 0.43100620858934935, 0.1942377614324795, 0.4906767050383777, 0.3593540329557037, 0.48897054500140047, 0.8870083684438635, 0.7565212172893266, 0.3407065583321811, 0.488499671911241, 0.9044453170520311, 0.48288349145942666, 0.9004944454982946, 0.7270418355049255, 0.6012667038332827, 0.7730250244329887, 0.37163333178338687, 0.8384479350365978, 0.3227131612319124, 0.14839684092546268, 0.8839836045190522, 0.03855445637631527, 0.9030950301986007, 0.40089074949874715, 0.7167611463454734, 0.6841749573406255, 0.7006654935882408, 0.4899905792075667, 0.952645805061099, 0.1660817878999985, 0.08713935237358417, 0.6360385466573328, 0.45814612815274647, 0.36079817844276396, 0.8191002831743147, 0.7758734542679913, 0.3672170939726287, 0.0821983266239128, 0.22735743708316547, 0.8827612011366579, 0.16294969375309276, 0.14316790725805306, 0.7687903053153128, 0.8903662420051904, 0.6363700069958681, 0.5495154296335041, 0.646974854219965, 0.1662292201142156, 0.5403983128436651, 0.5567880865815498, 0.7582734316837413, 0.32274358033832196, 0.7899177567237766, 0.14593021366890624, 0.07109721800915447, 0.6269757409404126, 0.17707404735022803, 0.2852211621337746, 0.222855398771846, 0.17278128495962541, 0.6642130026942895, 0.5669389656045777, 0.10602923959358668, 0.9306663548005181, 0.4354441303028146, 0.49925775217499657, 0.9714773884995452, 0.46933886543803527, 0.5732482107528604, 0.9782903514801252, 0.003617422213274879, 0.20823596368085073, 0.5659542265478954, 0.28044990670600245, 0.4118584164660316, 0.5280349284095007, 0.7819688461857907, 0.008380148162595913, 0.7582974035857967, 0.692274295711465, 0.7033964209970381, 0.459039438403936, 0.9266261698770437, 0.9814608746090258, 0.12505119031675438, 0.4989674502999548, 0.15456852335694915, 0.7050829929305539, 0.4899658898264858, 0.3061535534949438, 0.21886717761883268, 0.6801011115492205, 0.7675803672595314, 0.9574712562720578, 0.9595875812820007, 0.6226462624521372, 0.4402989055881257, 0.13957480751006068, 0.015355489244549103, 0.8844165918706132, 0.8979300997961542, 0.18554033332209408, 0.7975205438559934, 0.4092694236254455, 0.24336782553172565, 0.7314090130176317, 0.4970151728232043, 0.029046553878170567, 0.8112065726283106, 0.940412585561525, 0.31226378429034163, 0.07794983239732078, 0.1276035017511732, 0.842121644095542, 0.9522770401375675, 0.5249012999103058, 0.05222741203109382, 0.0376946173574082, 0.5814869978084197, 0.7293126965261645, 0.26160906359724645, 0.8347049466645863, 0.5213108737300144, 0.6466875479774858, 0.21392240372150728, 0.5433641297353714, 0.03634276608163012, 0.7486764933530231, 0.5390941703655878, 0.41616651900421975, 0.6941222600342826, 0.41411016289624425, 0.03685362332803499, 0.8120202672487916, 0.5653399476146768, 0.5241403205735731, 0.72890681477522, 0.17283083642552577, 0.2144684821586762, 0.7630846000210761, 0.19325190518333335, 0.5997257703308343, 0.7478149991500123, 0.6221762717950524, 0.8213465280915754, 0.8564581799303568, 0.37262439822934557, 0.620238443748559, 0.9707489305485756, 0.24623341455354375, 0.5915813802504658, 0.29728066740747483, 0.765982546922249, 0.8516318311912112, 0.2175074213310697, 0.30634089520967034, 0.13670729630039624, 0.4104717105140029, 0.5702862339358866, 0.48781551979513493, 0.12101196523494917, 0.7204106487547365, 0.12422528252050158, 0.21914835094859098, 0.9100671749228464, 0.38780279396955886, 0.15080346723176885, 0.6508717091666221, 0.17544385348156966, 0.7062365834322223, 0.4651903729136251, 0.4400026206105151, 0.2595712176016646, 0.19820521690392634, 0.6022809211704685, 0.06499498735521692, 0.4977741502307016, 0.09580420434529036, 0.6229068251846043, 0.3997625585981035, 0.5554069557446442, 0.09123401959721145, 0.4555962644652034, 0.4618537279540924, 0.4232696425090142, 0.7661994649984093, 0.36711484750845025, 0.6602129642617525, 0.5856656011548966, 0.24544516009436712, 0.9017361927825662, 0.7836110474274166, 0.778134636691487, 0.4689096304125806, 0.5653831890329833, 0.8472757912569392, 0.4412081792590239, 0.38713456827135084, 0.5881751295638957, 0.49419219847348206, 0.7965104303337288, 0.009658281061194818, 0.3309092542635881, 0.4116572770642375, 0.5740353998112648, 0.7876869747201949, 0.6184319508158961, 0.44689602459368294, 0.9949804293485819, 0.6623113547406154, 0.16690903934100543, 0.8873912750519208, 0.21047261496323133, 0.9126729380040393, 0.5437968705649276, 0.8547800856022514, 0.9753139370036432, 0.6545396101110816, 0.18416376983960203, 0.4652961990688069, 0.23636159801359913, 0.3161222463603781, 0.9405206995852127, 0.35209150546873413, 0.2612357617354647, 0.7669681666874368, 0.5622426388769249, 0.6885052454779612, 0.8038717467278913, 0.42806987836722166, 0.5390946834235756, 0.8241268329065514, 0.2816662053766853, 0.43271601961198536, 0.3546882973933174, 0.3123400679546886, 0.8044024797504884, 0.18385151432780056, 0.09742568195151846, 0.6680612345028595, 0.8130779632000169, 0.12210184219389586, 0.8708705853013151, 0.1992081510803403, 0.1415870582106824, 0.30416454582716146, 0.7510284252137773, 0.04340643695169111, 0.48328703186421706, 0.9428291493202412, 0.637101722457211, 0.45099331679055343, 0.7113692215367728, 0.7837691892199189, 0.16190785793886509, 0.7825506969838596, 0.8460902383418621, 0.7920174634843665, 0.6793656147845987, 0.4705877707267101, 0.6983892885007057, 0.609997096050793, 0.42752711436545376, 0.36998370863064556, 0.5349849355412531, 0.7126197850413926, 0.3450399563930353, 0.5904074978899103, 0.617560371092591, 0.8437046987077099, 0.27341675229463647, 0.2645368134759384, 0.7160450524784653, 0.3260937659692926, 0.4868125935793397, 0.5264433918866334, 0.4141192639101384, 0.9460034289758379, 0.6027502510856219, 0.6300307397504198, 0.48466019883526135, 0.5551198494078631, 0.6701475449534646, 0.9012075484916191, 0.3523839969354364, 0.48464149060223305, 0.9480584268335772, 0.02123841955830097, 0.2190582005435352, 0.48429441296479814, 0.9144869020048872, 0.6925776753344967, 0.57216124866501, 0.16110327365418653, 0.13190604027934372, 0.055115101900676255, 0.9977412571654896, 0.7502522088598729, 0.2217384081739373, 0.4502981391167644, 0.1143733729551295, 0.15071866642328446, 0.28730076130215687, 0.30722877395995196, 0.5807630920273893, 0.7743856863501432, 0.35073664417454486, 0.07608957426726237, 0.3242599758963747, 0.1959711425296249, 0.8678327239191715, 0.6987246142582734, 0.26655663376811134, 0.03214428222962662, 0.9163496163970758, 0.8778864759838196, 0.6682857444532649, 0.2520561103755973, 0.9353840852946753, 0.5009977390064025, 0.3390214585019413, 0.34496204202791914, 0.009749477293079178, 0.1750806958823562, 0.7541231253925336, 0.7807875314752765, 0.07671435088802159, 0.2952035321407428, 0.3928474396168219, 0.6384562056506056, 0.9306917964135548, 0.29022989347758676, 0.1752409705163681, 0.3615984989882929, 0.5042316137649276, 0.9307855640879155, 0.33369095152858363, 0.5889972073523481, 0.07088144702777288, 0.31725963777099797, 0.15226197134132213, 0.8854157553289445, 0.6255879657493837, 0.9850325688314581, 0.738868124986551, 0.37568410170058375, 0.7003484392737419, 0.8573254361456892, 0.09509915918999612, 0.9273776438483797, 0.7873650485569003, 0.7848026547347504, 0.7673908344529615, 0.6118103325222369, 0.7500465564704132, 0.15529847200881752, 0.026808696077421867, 0.5729200819436895, 0.14756026223529684, 0.55753319582846, 0.7743192446180736, 0.3711252932075446, 0.5862540676326985, 0.9245274010410062, 0.3206707481306149, 0.6137278534568711, 0.24298095056847924, 0.6448994129170375, 0.46968242021993944, 0.3206871139588491, 0.7019943614715987, 0.04617008059624195, 0.09564851053600343, 0.375592859389856, 0.7175925912139642, 0.27837235740280186, 0.03561612203895759, 0.6301986150913171, 0.9850418439434245, 0.5233837002917712, 0.9420932398652992, 0.6801998669384548, 0.46050194737624006, 0.7431194634704841, 0.9261738591850471, 0.8997751387269006, 0.6358931136068172, 0.1578161946770682, 0.9498500800072147, 0.555763616593164, 0.6032944869038372, 0.954589168027848, 0.37264089817546653, 0.10774475847924747, 0.17585347268591467, 0.5034719591261876, 0.953068583450757, 0.6858447840867116, 0.5920783257533169, 0.800802317541777, 0.4969294224105828, 0.8327021964670941, 0.3489281965467961, 0.5876460810026024, 0.018631124721412484, 0.9762735705388261, 0.3450377539317617, 0.8664995619817126, 0.8494179052441462, 0.46585971792933334, 0.2990981560727781, 0.34059999748178216, 0.6454222197850171, 0.44168410103360867, 0.056660444734211834, 0.8876196012100044, 0.38069476446541695, 0.04937609353482497, 0.5220555279385446, 0.315299794071894, 0.6685897827713608, 0.8397624268333489, 0.9556220124804904, 0.29205447601253187, 0.764673807209028, 0.7398952231921495, 0.6232961512951933, 0.2712146795815681, 0.20243803284436235, 0.8602847470194057, 0.15962223411230725, 0.4133412603684523, 0.39447416461257323, 0.12191996682119233, 0.8867623240411905, 0.8485911187829595, 0.28059514449258616, 0.2222973556638188, 0.4575033234670913, 0.4255589274983831, 0.2757170275647197, 0.9732212348893373, 0.7595009919154937, 0.19917751741560563, 0.5928622699284407, 0.6280112557323658, 0.6902125667980579, 0.6632922074504858, 0.3918548403358988, 0.20422075722228683, 0.330686322127939, 0.5188381995184453, 0.1559259970522583, 0.39129655957184994, 0.9887854558595667, 0.07513679627328107, 0.7396799790343711, 0.5580736012264869, 0.06467918251400573, 0.7935662365812974, 0.7687971746421685, 0.7103078495285694, 0.10195354796927303, 0.09980131662314018, 0.4502724736948077, 0.9793689195890094, 0.1119081814931987, 0.21616264537224605, 0.2720264268729303, 0.8536239294533816, 0.6765764880062813, 0.42942242126580077, 0.906867762851367, 0.44645897576114824, 0.6051447885841313, 0.7437215167545904, 0.0156672072105426, 0.33957340569682704, 0.19286375234643494, 0.529107387532468, 0.5679788614636623, 0.8770394596476377, 0.4810125905516983, 0.8224993538215689]

########################################################################################################################
# Calling function NGnet to create two NGnets
# There are different variants for quick check of final NGnets:
# # NGnet1:
# NGnet1 = create_NGnet("1", sigma, Overlap, G_Din, G_Dout,w1)
# NGnet1 = create_NGnet("1", 2, 0.3, 20, 100, w1)
# NGnet1 = create_NGnet("1", 1.2, 0.2, 40, 90, w1)
NGnet1 = create_NGnet("1", 0.8, 0.25, 38, 90, w1)  # Number_of_Gauss: 78
# # NGnet2:
# NGnet2 = create_NGnet("2", 2.5, +0.1, 30, 90, w2)
NGnet2 = create_NGnet("2", 1, +0.1, 20, 90,w2)

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
