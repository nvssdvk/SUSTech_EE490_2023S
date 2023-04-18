import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cst.interface

line_break = '\n'


def add_to_history(modeler, header, vbacode):
    res = modeler.add_to_history(header, vbacode)
    return res


def define_material(modeler, name='', epsilon=1.24):
    header = "define material: material1/{:s}".format(name)
    vbacode = [
        'With Material ',
        '     .Reset ',
        '     .Name "{:s}"'.format(name),
        '     .Folder "material1"',
        '     .Rho "0.0"',
        '     .ThermalType "Normal"',
        '     .ThermalConductivity "0"',
        '     .SpecificHeat "0", "J/K/kg"',
        '     .DynamicViscosity "0"',
        '     .Emissivity "0"',
        '     .MetabolicRate "0.0"',
        '     .VoxelConvection "0.0"',
        '     .BloodFlow "0"',
        '     .MechanicsType "Unused"',
        '     .FrqType "all"',
        '     .Type "Normal"',
        '     .MaterialUnit "Frequency", "GHz"',
        '     .MaterialUnit "Geometry", "mm"',
        '     .MaterialUnit "Time", "ns"',
        '     .MaterialUnit "Temperature", "Kelvin"',
        '     .Epsilon "{:.2f}"'.format(epsilon),
        '     .Mu "1"',
        '     .Sigma "0"',
        '     .TanD "0.0"',
        '     .TanDFreq "0.0"',
        '     .TanDGiven "False"',
        '     .TanDModel "ConstTanD"',
        '     .EnableUserConstTanDModelOrderEps "False"',
        '     .ConstTanDModelOrderEps "1"',
        '     .SetElParametricConductivity "False"',
        '     .ReferenceCoordSystem "Global"',
        '     .CoordSystemType "Cartesian"',
        '     .SigmaM "0"',
        '     .TanDM "0.0"',
        '     .TanDMFreq "0.0"',
        '     .TanDMGiven "False"',
        '     .TanDMModel "ConstTanD"',
        '     .EnableUserConstTanDModelOrderMu "False"',
        '     .ConstTanDModelOrderMu "1"',
        '     .SetMagParametricConductivity "False"',
        '     .DispModelEps  "None"',
        '     .DispModelMu "None"',
        '     .DispersiveFittingSchemeEps "Nth Order"',
        '     .MaximalOrderNthModelFitEps "10"',
        '     .ErrorLimitNthModelFitEps "0.1"',
        '     .UseOnlyDataInSimFreqRangeNthModelEps "False"',
        '     .DispersiveFittingSchemeMu "Nth Order"',
        '     .MaximalOrderNthModelFitMu "10"',
        '     .ErrorLimitNthModelFitMu "0.1"',
        '     .UseOnlyDataInSimFreqRangeNthModelMu "False"',
        '     .UseGeneralDispersionEps "False"',
        '     .UseGeneralDispersionMu "False"',
        '     .NLAnisotropy "False"',
        '     .NLAStackingFactor "1"',
        '     .NLADirectionX "1"',
        '     .NLADirectionY "0"',
        '     .NLADirectionZ "0"',
        '     .Colour "0", "1", "1" ',
        '     .Wireframe "False" ',
        '     .Reflection "False" ',
        '     .Allowoutline "True" ',
        '     .Transparentoutline "False" ',
        '     .Transparency "0" ',
        '     .Create',
        'End With'
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


def set_units(modeler):
    header = "set units"
    vbacode = [
        'With Units',
        '    .SetUnit "Length", "mm"',
        '    .SetUnit "Frequency", "GHz"',
        '    .SetUnit "Voltage", "V"',
        '    .SetUnit "Resistance", "Ohm"',
        '    .SetUnit "Inductance", "nH"',
        '    .SetUnit "Temperature",  "degC"',
        '    .SetUnit "Time", "ns"',
        '    .SetUnit "Current", "A"',
        '    .SetUnit "Conductance", "S"',
        '    .SetUnit "Capacitance", "pF"',
        'End With'
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


def set_freq_range(modeler, f_min, f_max):
    header = "set frequency range"
    vbacode = 'Solver.FrequencyRange "{:.1f}", "{:.1f}"'.format(f_min, f_max)
    add_to_history(modeler, header, vbacode)


def set_background(modeler, zmax=0.0):
    header = "define background"
    vbacode = [
        'With Background',
        '     .Type "Normal"',
        '     .Epsilon "1.0"',
        '     .Mu "1.0"',
        '     .Rho "1.204"',
        '     .ThermalType "Normal"',
        '     .ThermalConductivity "0.026"',
        '      .SpecificHeat "1005", "J/K/kg"',
        '     .XminSpace "0.0"',
        '     .XmaxSpace "0.0"',
        '     .YminSpace "0.0"',
        '     .YmaxSpace "0.0"',
        '     .ZminSpace "0.0"',
        '     .ZmaxSpace "{:.2f}"'.format(zmax),
        'End With',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


def set_boundaries(modeler):
    header = "define boundaries"
    vbacode = [
        'With Boundary',
        '     .Xmin "expanded open"',
        '     .Xmax "expanded open"',
        '     .Ymin "expanded open"',
        '     .Ymax "expanded open"',
        '     .Zmin "expanded open"',
        '     .Zmax "expanded open"',
        '     .Xsymmetry "none"',
        '     .Ysymmetry "none"',
        '     .Zsymmetry "none"',
        '     .ApplyInAllDirections "False"',
        '     .OpenAddSpaceFactor "0.5"',
        '     .XPeriodicShift "0.0"',
        '     .YPeriodicShift "0.0"',
        '     .ZPeriodicShift "0.0"',
        '     .PeriodicUseConstantAngles "False"',
        '     .SetPeriodicBoundaryAngles "theta", "phi"',
        '     .SetPeriodicBoundaryAnglesDirection "inward"',
        '     .UnitCellFitToBoundingBox "True"',
        '     .UnitCellDs1 "0.0"',
        '     .UnitCellDs2 "0.0"',
        '     .UnitCellAngle "90.0"',
        'End With',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


def zoom(modeler):
    header = "zoom"
    vbacode = 'Plot.ZoomToStructure'
    add_to_history(modeler, header, vbacode)


def modeling_unit_main(row, col, position_bottom_center, a, b, h, e):
    x, y, z = position_bottom_center
    vbacode = [
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "bottom" ',
        '     .Curve "curve_{:d}_{:d}" '.format(row, col),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x - b / 2, y - b / 2, z),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x + b / 2, y - b / 2, z),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x + b / 2, y + b / 2, z),
        '     .Point "{:.2f}", "{:.2f}","{:.2f}" '.format(x - b / 2, y + b / 2, z),
        '     .Point "{:.2f}", "{:.2f}","{:.2f}" '.format(x - b / 2, y - b / 2, z),
        '     .Create ',
        'End With',
        '',
        '',
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "top" ',
        '     .Curve "curve_{:d}_{:d}" '.format(row, col),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x - a / 2, y - a / 2, z + h),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x + a / 2, y - a / 2, z + h),
        '     .Point "{:.2f}", "{:.2f}", "{:.2f}" '.format(x + a / 2, y + a / 2, z + h),
        '     .Point "{:.2f}", "{:.2f}","{:.2f}" '.format(x - a / 2, y + a / 2, z + h),
        '     .Point "{:.2f}", "{:.2f}","{:.2f}" '.format(x - a / 2, y - a / 2, z + h),
        '     .Create ',
        'End With',
        '',
        '',
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid_bottom" ',
        '     .Component " component_{:d}_{:d}" '.format(row, col),
        '     .Material "material1/{:s}" '.format('pla_{:.2f}'.format(e)),
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve " curve_{:d}_{:d}:bottom" '.format(row, col),
        '     .Create',
        'End With',
        '',
        '',
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid_top" ',
        '     .Component " component_{:d}_{:d}" '.format(row, col),
        '     .Material "material1/{:s}" '.format('pla_{:.2f}'.format(e)),
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve " curve_{:d}_{:d}:top" '.format(row, col),
        '     .Create',
        'End With',
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:.2f}", "{:.2f}", "{:.2f}"'.format(
            "component_{:d}_{:d}:solid_bottom".format(row, col), x, y, z),
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:.2f}", "{:.2f}", "{:.2f}"'.format(
            "component_{:d}_{:d}:solid_top".format(row, col), x, y, z + h),
        '',
        '',
        'With Loft ',
        '     .Reset ',
        '     .Name "cell" ',
        '     .Component "component_{:d}_{:d}" '.format(row, col),
        '     .Material "material1/{:s}" '.format('pla_{:.2f}'.format(e)),
        '     .Tangency "0.0" ',
        '     .Minimizetwist "true" ',
        '     .CreateNew ',
        'End With',
        '',
        '',
        'Solid.Add "{:s}", "{:s}"'.format("component_{:d}_{:d}:cell".format(row, col),
                                          "component_{:d}_{:d}:solid_bottom".format(row, col)),
        '',
        '',
        'Solid.Add "{:s}", "{:s}"'.format("component_{:d}_{:d}:cell".format(row, col),
                                          "component_{:d}_{:d}:solid_top".format(row, col)),
        '',
        '',
    ]
    # vbacode = line_break.join(vbacode)
    return vbacode


if __name__ == "__main__":
    # df_name = ['row', 'col', 'phase', 'phase_loss', 'weight', 'score', 'a', 'h', 'e']
    data_para = pd.read_csv(r"../../data/dataset/aperture_para.csv", header=0, engine="c").values
    aperture_para = data_para[:, -3:]
    e_range = np.unique(aperture_para[:, 2])

    my_project_name = f'aperture.cst'
    my_project_path = os.path.abspath(f'../../cst/{my_project_name}')
    my_de = cst.interface.DesignEnvironment()
    my_mws = my_de.new_mws()
    my_mws.save(my_project_path, include_results=False)
    my_modeler = my_mws.modeler

    set_units(my_modeler)
    set_freq_range(my_modeler, 6, 14)
    set_background(my_modeler)

    for e in e_range:
        define_material(my_modeler, name='pla_{:.2f}'.format(e), epsilon=e)

    unit_num = int(np.sqrt(aperture_para.shape[0]))
    unit_len = 15
    dx = dy = unit_len
    x_arr = np.arange(-unit_num / 2 * dx + unit_len / 2, unit_num / 2 * dx + unit_len / 2, dx)
    y_arr = np.arange(-unit_num / 2 * dy + unit_len / 2, unit_num / 2 * dy + unit_len / 2, dy)
    xx, yy = np.meshgrid(x_arr, y_arr)

    vbacode = ['']
    cnt = 0
    for i in range(unit_num):
        # time.sleep(10)
        for j in range(unit_num):
            b = 15
            a, h, e = aperture_para[cnt, :]
            position_bottom_center = [xx[i, j], yy[i, j], 0]
            temp_vbacode = modeling_unit_main(i, j, position_bottom_center, a, b, h, e)
            vbacode += temp_vbacode
            cnt += 1
    header = "modeling aperture"
    vbacode = line_break.join(vbacode)

    time_start = time.time()
    add_to_history(my_modeler, header, vbacode)
    time_end = time.time()
    print("Modeling the aperture, the total time is {:.1f} seconds".format(time_end - time_start), end='\r')
    my_mws.save()
