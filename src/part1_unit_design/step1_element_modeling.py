import datetime
import os

import cst.interface

line_break = '\n'

if __name__ == '__main__':
    my_project_name = f'part1_unit_design.cst'
    my_project_path = os.path.abspath(f'../../cst/{my_project_name}')
    my_de = cst.interface.DesignEnvironment()
    my_mws = my_de.new_mws()
    my_mws.save(my_project_path)

    my_modeler = my_mws.modeler


    def add_to_history(header, vbacode):
        res = my_modeler.add_to_history(header, vbacode)
        return res


    def add_para(param_name, para_value):
        header = f'store a parameter'
        vba_code = f'MakeSureParameterExists("{param_name}", "{para_value}")'
        res = add_to_history(header, vba_code)
        return res


    header = "use template: FSS"
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

        'ThermalSolver.AmbientTemperature "0"',

        'Solver.FrequencyRange "6", "14"',

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
        '     .ZmaxSpace "0.0"',
        'End With',

        'With FloquetPort',
        '     .Reset',
        '     .SetDialogTheta "0"',
        '     .SetDialogPhi "0"',
        '     .SetSortCode "+beta/pw"',
        '     .SetCustomizedListFlag "False"',
        '     .Port "Zmin"',
        '     .SetNumberOfModesConsidered "2"',
        '     .Port "Zmax"',
        '     .SetNumberOfModesConsidered "2"',
        'End With',

        'MakeSureParameterExists "theta", "0"',
        'SetParameterDescription "theta", "spherical angle of incident plane wave"',
        'MakeSureParameterExists "phi", "0"',
        'SetParameterDescription "phi", "spherical angle of incident plane wave"',

        'With Boundary',
        '     .Xmin "unit part1_unit_design"',
        '     .Xmax "unit part1_unit_design"',
        '     .Ymin "unit part1_unit_design"',
        '     .Ymax "unit part1_unit_design"',
        '     .Zmin "expanded open"',
        '     .Zmax "expanded open"',
        '     .Xsymmetry "none"',
        '     .Ysymmetry "none"',
        '     .Zsymmetry "none"',
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

        'With Mesh',
        '     .MeshType "Tetrahedral"',
        'End With',

        'With FDSolver',
        '     .Reset',
        '     .Stimulation "List", "List"',
        '     .ResetExcitationList',
        '     .AddToExcitationList "Zmax", "TE(0,0);TM(0,0)"',
        '     .LowFrequencyStabilization "False"',
        'End With',

        'With MeshSettings',
        '     .SetMeshType "Tet"',
        '     .Set "Version", 1%',
        'End With',

        'With Mesh',
        '     .MeshType "Tetrahedral"',
        'End With',

        'ChangeSolverType("HF Frequency Domain")',

    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    add_para(f'lambda', f'300/11')
    add_para(f'b', f'14')
    add_para(f'a', f'2')
    add_para(f'e', f'2.7')
    add_para(f'h', f'2')
    add_para(f'hf', f'lambda/2')

    header = "define material: PLA/material1"
    vbacode = [
        'With Material ',
        '     .Reset ',
        '     .Name "material1"',
        '     .Folder "PLA"',
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
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define curve 3dpolygon: curve1:buttom"
    vbacode = [
        'With Polygon3D ',
        '     .Reset ',
        '     .Version 10 ',
        '     .Name "buttom" ',
        '     .Curve "curve1" ',
        '     .Point "-b/2", "-b/2", "0" ',
        '     .Point "b/2", "-b/2", "0" ',
        '     .Point "b/2", "b/2", "0" ',
        '     .Point "-b/2", "b/2", "0" ',
        '     .Point "-b/2", "-b/2", "0" ',
        '     .Create ',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define curve 3dpolygon: curve1:top"
    vbacode = [
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
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    # header = "new component: componenti"
    # vbacode = [
    #     'Component.New "component1"'
    # ]
    # add_to_history(header, vbacode)

    header = "define extrudeprofile: component1:solid3"
    vbacode = [
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid3" ',
        '     .Component "component1" ',
        '     .Material "PLA/material1" ',
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve "curve1:buttom" ',
        '     .Create',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define extrudeprofile: component1:solid4"
    vbacode = [
        'With ExtrudeCurve',
        '     .Reset ',
        '     .Name "solid4" ',
        '     .Component "component1" ',
        '     .Material "PLA/material1" ',
        '     .Thickness "0.0" ',
        '     .Twistangle "0.0" ',
        '     .Taperangle "0.0" ',
        '     .DeleteProfile "True" ',
        '     .Curve "curve1:top" ',
        '     .Create',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "pick face"
    vbacode = 'Pick.PickFaceFromPoint  "component1:solid3", "0", "0", "0"'
    add_to_history(header, vbacode)

    header = "pick face"
    vbacode = 'Pick.PickFaceFromPoint "component1:solid4", "0", "0", "h"'
    add_to_history(header, vbacode)

    header = "define loft: component1:part1_unit_design"
    vbacode = [
        'With Loft ',
        '     .Reset ',
        '     .Name "part1_unit_design" ',
        '     .Component "component1" ',
        '     .Material "PLA/material1" ',
        '     .Tangency "0.0" ',
        '     .Minimizetwist "true" ',
        '     .CreateNew ',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "boolean add shapes: component1:solid3, componenti:part1_unit_design"
    vbacode = 'Solid.Add "component1:solid3", "component1:part1_unit_design"'
    add_to_history(header, vbacode)

    header = "boolean add shapes: component1:solid4, component1:solid3"
    vbacode = 'Solid.Add "component1:solid4", "component1:solid3"'
    add_to_history(header, vbacode)

    header = "rename block: component1:solid4 to: component1:part1_unit_design"
    vbacode = 'Solid.Rename "component1:solid4", "part1_unit_design"'
    add_to_history(header, vbacode)

    header = "define frequency range"
    vbacode = 'Solver.FrequencyRange "6", "14"'
    add_to_history(header, vbacode)

    header = "define boundaries"
    vbacode = [
        'With Boundary',
        '     .Xmin "unit part1_unit_design"',
        '     .Xmax "unit part1_unit_design"',
        '     .Ymin "unit part1_unit_design"',
        '     .Ymax "unit part1_unit_design"',
        '     .Zmin "electric"',
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
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define background"
    vbacode = [
        'With Background ',
        '     .ResetBackground ',
        '     .XminSpace "0.0" ',
        '     .XmaxSpace "0.0" ',
        '     .YminSpace "0.0" ',
        '     .YmaxSpace "0.0" ',
        '     .ZminSpace "0.0" ',
        '     .ZmaxSpace "hf" ',
        '     .ApplyInAllDirections "False" ',
        'End With ',
        'With Material ',
        '     .Reset ',
        '     .Rho "1.204"',
        '     .ThermalType "Normal"',
        '     .ThermalConductivity "0.026"',
        '     .SpecificHeat "1004.9999999999999", "J/K/kg"',
        '     .DynamicViscosity "0"',
        '     .Emissivity "0"',
        '     .MetabolicRate "0.0"',
        '     .VoxelConvection "0.0"',
        '     .BloodFlow "0"',
        '     .MechanicsType "Unused"',
        '     .FrqType "all"',
        '     .Type "Normal"',
        '     .MaterialUnit "Frequency", "Hz"',
        '     .MaterialUnit "Geometry", "m"',
        '     .MaterialUnit "Time", "s"',
        '     .MaterialUnit "Temperature", "Kelvin"',
        '     .Epsilon "1.0"',
        '     .Mu "1.0"',
        '     .Sigma "0"',
        '     .TanD "0.0"',
        '     .TanDFreq "0.0"',
        '     .TanDGiven "False"',
        '     .TanDModel "ConstSigma"',
        '     .EnableUserConstTanDModelOrderEps "False"',
        '     .ConstTanDModelOrderEps "1"',
        '     .SetElParametricConductivity "False"',
        '     .ReferenceCoordSystem "Global"',
        '     .CoordSystemType "Cartesian"',
        '     .SigmaM "0"',
        '     .TanDM "0.0"',
        '     .TanDMFreq "0.0"',
        '     .TanDMGiven "False"',
        '     .TanDMModel "ConstSigma"',
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
        '     .Colour "0.6", "0.6", "0.6" ',
        '     .Wireframe "False" ',
        '     .Reflection "False" ',
        '     .Allowoutline "True" ',
        '     .Transparentoutline "False" ',
        '     .Transparency "0" ',
        '     .ChangeBackgroundMaterial',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define Floquet port boundaries"
    vbacode = [
        'With FloquetPort',
        '     .Reset',
        '     .SetDialogTheta "0" ',
        '     .SetDialogPhi "0" ',
        '     .SetPolarizationIndependentOfScanAnglePhi "0.0", "False"  ',
        '     .SetSortCode "+beta/pw" ',
        '     .SetCustomizedListFlag "False" ',
        '     .Port "Zmax" ',
        '     .SetNumberOfModesConsidered "2" ',
        '     .SetDistanceToReferencePlane "-hf" ',
        '     .SetUseCircularPolarization "False" ',
        'End With',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    header = "define frequency domain solver acceleration"
    vbacode = [
        'With FDSolver ',
        '     .MPIParallelization "False"',
        '     .UseDistributedComputing "False"',
        '     .NetworkComputingStrategy "RunRemote"',
        '     .NetworkComputingJobCount "3"',
        '     .UseParallelization "True"',
        '     .MaxCPUs "1024"',
        '     .MaximumNumberOfCPUDevices "8"',
        'End With',
        'With MeshSettings',
        '     .SetMeshType "Unstr"',
        '     .Set "UseDC", "0"',
        'End With',
        'UseDistributedComputingForParameters "False"',
        'MaxNumberOfDistributedComputingParameters "2"',
        'UseDistributedComputingMemorySetting "False"',
        'MinDistributedComputingMemoryLimit "0"',
        'UseDistributedComputingSharedDirectory "False"',
        '',
        '',
    ]
    vbacode = line_break.join(vbacode)
    add_to_history(header, vbacode)

    # my_modeler.run_solver()

    my_mws.save()

    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
    #
    # header = ""
    # vbacode = [
    #
    # ]
    # add_to_history(header, vbacode)
