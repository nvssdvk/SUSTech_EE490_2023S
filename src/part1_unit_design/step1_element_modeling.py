import datetime
import os

import cst.interface

line_break = '\n'


def add_to_history(modeler, header, vbacode):
    res = modeler.add_to_history(header, vbacode)
    return res


def add_para(modeler, param_name, para_value):
    header = f'store a parameter'
    vbacode = f'MakeSureParameterExists("{param_name}", "{para_value}")'
    res = add_to_history(modeler, header, vbacode)
    return res


def modling_main(modeler):
    header = "modling"
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
        'End With',
        '',
        '',
        'Solver.FrequencyRange "6", "14"',
        '',
        '',
        'With Material ',
        '     .Reset ',
        '     .Name "pla"',
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
        '     .Epsilon "e"',
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
        'End With',
        '',
        '',
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "bottom" ',
        '     .Curve "curve1" ',
        '     .Point "-b/2", "-b/2", "0" ',
        '     .Point "b/2", "-b/2", "0" ',
        '     .Point "b/2", "b/2", "0" ',
        '     .Point "-b/2", "b/2", "0" ',
        '     .Point "-b/2", "-b/2", "0" ',
        '     .Create ',
        'End With',
        '',
        '',
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "top" ',
        '     .Curve "curve1" ',
        '     .Point "-a/2", "-a/2", "h" ',
        '     .Point "a/2", "-a/2", "h" ',
        '     .Point "a/2", "a/2", "h" ',
        '     .Point "-a/2", "a/2", "h" ',
        '     .Point "-a/2", "-a/2", "h" ',
        '     .Create ',
        'End With',
        '',
        '',
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid3" ',
        '     .Component "component1" ',
        '     .Material "material1/pla" ',
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve "curve1:bottom" ',
        '     .Create',
        'End With',
        '',
        '',
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid4" ',
        '     .Component "component1" ',
        '     .Material "material1/pla" ',
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve "curve1:top" ',
        '     .Create',
        'End With',
        '',
        '',
        'Pick.PickFaceFromPoint  "component1:solid3", "0", "0", "0"',
        '',
        '',
        'Pick.PickFaceFromPoint "component1:solid4", "0", "0", "h"',
        '',
        '',
        'With Loft ',
        '     .Reset ',
        '     .Name "cell" ',
        '     .Component "component1" ',
        '     .Material "material1/pla" ',
        '     .Tangency "0.0" ',
        '     .Minimizetwist "true" ',
        '     .CreateNew ',
        'End With',
        '',
        '',
        'Solid.Add "component1:cell", "component1:solid3"',
        '',
        '',
        'Solid.Add "component1:cell", "component1:solid4"',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(modeler, header, vbacode)


if __name__ == '__main__':
    my_project_name = f'unit.cst'
    my_project_path = os.path.abspath(f'../../cst/{my_project_name}')
    my_de = cst.interface.DesignEnvironment()
    my_mws = my_de.new_mws()
    my_mws.save(my_project_path)

    my_modeler = my_mws.modeler

    add_para(my_modeler, f'lambda', f'300/11')
    add_para(my_modeler, f'b', f'14')
    add_para(my_modeler, f'a', f'2')
    add_para(my_modeler, f'e', f'2.7')
    add_para(my_modeler, f'h', f'2')
    add_para(my_modeler, f'hf', f'lambda/2')

    modling_main(my_modeler)

    my_mws.save()
