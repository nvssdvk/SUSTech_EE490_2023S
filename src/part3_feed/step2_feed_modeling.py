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


def set_background(modeler, zmax='zmax'):
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
        '     .ZmaxSpace "{:s}"'.format(zmax),
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


def modeling_horn_main(modeler):
    header = 'modeling'
    vbacode = [
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "aperture_top" ',
        '     .Curve "curve1" ',
        '     .Point "{:s}", "{:s}", "{:s}" '.format('-a / 2', '-b / 2',' wg_h + h'),
        '     .Point "{:s}", "{:s}", "{:s}" '.format('a / 2', '- b / 2', 'wg_h + h'),
        '     .Point "{:s}", "{:s}", "{:s}" '.format('a / 2', 'b / 2',' wg_h + h'),
        '     .Point "{:s}", "{:s}","{:s}" '.format('-a / 2', 'b / 2', 'wg_h + h'),
        '     .Point "{:s}", "{:s}","{:s}" '.format('-a / 2', '- b / 2', 'wg_h + h'),
        '     .Create ',
        'End With',
        '',
        '',
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "aperture_bottom" ',
        '     .Curve "curve1" ',
        '     .Point "{:s}", "{:s}", "{:s}" '.format('-wg_a / 2', '-wg_b / 2', 'wg_h'),
        '     .Point "{:s}", "{:s}", "{:s}" '.format('wg_a / 2', '- wg_b / 2', 'wg_h'),
        '     .Point "{:s}", "{:s}", "{:s}" '.format('wg_a / 2', 'wg_b / 2', 'wg_h'),
        '     .Point "{:s}", "{:s}","{:s}" '.format('-wg_a / 2', 'wg_b / 2', 'wg_h'),
        '     .Point "{:s}", "{:s}","{:s}" '.format('-wg_a / 2', '- wg_b / 2', 'wg_h'),
        '     .Create ',
        'End With',
        '',
        '',
        'With CoverCurve',
        '     .Reset ',
        '     .Name "aperture_bottom" ',
        '     .Component "component1" ',
        '     .Material "PEC" ',
        '     .Curve "curve1:aperture_bottom" ',
        '     .DeleteCurve "True" ',
        '     .Create',
        'End With',
        '',
        '',
        'With CoverCurve',
        '     .Reset ',
        '     .Name "aperture_top" ',
        '     .Component "component1" ',
        '     .Material "PEC" ',
        '     .Curve "curve1:aperture_top" ',
        '     .DeleteCurve "True" ',
        '     .Create',
        'End With',
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:s}", "{:s}", "{:s}"'.format(
            "component1:aperture_bottom", '0', '0', 'wg_h'),
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:s}", "{:s}", "{:s}"'.format(
            "component1:aperture_top",'0', '0', 'wg_h + h'),
        '',
        '',
        'With Loft ',
        '     .Reset ',
        '     .Name "aperture" ',
        '     .Component "component1" ',
        '     .Material "PEC" ',
        '     .Tangency "0.0" ',
        '     .Minimizetwist "true" ',
        '     .CreateNew ',
        'End With',
        '',
        '',
        'Solid.Add "{:s}", "{:s}"'.format("component1:aperture", "component1:aperture_bottom"),
        '',
        '',
        'Solid.Add "{:s}", "{:s}"'.format("component1:aperture", "component1:aperture_top"),
        '',
        '',
        'With Brick',
        '     .Reset ',
        '     .Name "waveguide" ',
        '     .Component "component1" ',
        '     .Material "PEC" ',
        '     .Xrange "{:s}", "{:s}" '.format('-wg_a / 2', 'wg_a / 2'),
        '     .Yrange "{:s}", "{:s}" '.format('-wg_b / 2', 'wg_b / 2'),
        '     .Zrange "{:s}", "{:s}" '.format('0', 'wg_h'),
        '     .Create',
        'End With',
        '',
        '',
        'Solid.Add "{:s}", "{:s}"'.format("component1:aperture", "component1:waveguide"),
        '',
        '',
        'Solid.Rename "component1:aperture", "horn"',
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:s}", "{:s}", "{:s}"'.format("component1:horn", '0', '0', '0'),
        '',
        '',
        'Pick.PickFaceFromPoint  "{:s}", "{:s}", "{:s}", "{:s}"'.format("component1:horn", '0', '0', 'wg_h + h'),
        '',
        '',
        'Solid.ShellAdvanced "component1:horn", "Inside", "{:s}", "True"'.format('thickness'),
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


if __name__ == "__main__":
    data = pd.read_csv(r'../../data/dataset/feed_horn_para.csv').values
    wg_a, wg_b, wg_h, a, b, h, thickness = data.squeeze()

    my_project_name = f'feed_horn_10GHz.cst'
    my_project_path = os.path.abspath(f'../../cst/{my_project_name}')
    my_de = cst.interface.DesignEnvironment()
    my_mws = my_de.new_mws()
    my_mws.save(my_project_path, include_results=True)
    my_modeler = my_mws.modeler


    def add_para(modeler, param_name, para_value):
        header = f'store a parameter'
        vbacode = f'MakeSureParameterExists("{param_name}", "{para_value}")'
        res = add_to_history(modeler, header, vbacode)
        return res


    add_para(my_modeler, f'wl', 30)
    add_para(my_modeler, f'zmax', 0)
    add_para(my_modeler, f'wg_a', wg_a)
    add_para(my_modeler, f'wg_b', wg_b)
    add_para(my_modeler, f'wg_h', wg_h)
    add_para(my_modeler, f'a', a)
    add_para(my_modeler, f'b', b)
    add_para(my_modeler, f'h', h)
    add_para(my_modeler, f'thickness', thickness)

    set_units(my_modeler)
    set_freq_range(my_modeler, 6, 14)
    set_background(my_modeler)

    modeling_horn_main(my_modeler)
    zoom(my_modeler)

    my_mws.save()
