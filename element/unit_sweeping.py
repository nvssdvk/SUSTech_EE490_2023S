import datetime
import os
import time
import numpy as np
import pandas as pd

import cst.interface
import cst.results

line_break = '\n'


def read_data():
    file_path = "../data/dataset/samples.csv"
    df = pd.read_csv(file_path, header=None, sep=',')
    df = np.asarray(df)
    num = len(df)

    ids = np.zeros((num, 1), dtype=int)
    a = np.zeros((num, 1), dtype=float)
    h = np.zeros((num, 1), dtype=float)
    e = np.zeros((num, 1), dtype=float)
    for i in range(num):
        ids[i] = df[i, 0]
        a[i] = df[i, 1]
        h[i] = df[i, 2]
        e[i] = df[i, 3]
    return num, ids, a, h, e


def add_to_history(modeler, header, vbacode):
    res = modeler.add_to_history(header, vbacode)
    return res


def add_para(modeler, param_name, para_value):
    header = f'store a parameter'
    vba_code = f'MakeSureParameterExists("{param_name}", "{para_value}")'
    res = add_to_history(modeler, header, vba_code)
    return res


def change_para(modeler, param_name, para_value):
    header = f'change a parameter'
    vba_code = f'StoreParameter ("{param_name}", "{para_value}")'
    res = add_to_history(modeler, header, vba_code)
    return res


if __name__ == '__main__':
    # my_project_path = os.path.abspath(r"D:\User\Course\EE490.2023S\src\data\RA_23_03_09\element.cst")
    my_project_path = r"D:\User\Course\EE490.2023S\SUSTech_EE490_2023S\cst\cell_v4.cst"

    num, ids, a, h, e = read_data()

    time_start = time.time()
    my_de = cst.interface.DesignEnvironment()
    my_mws = my_de.open_project(my_project_path)
    my_modeler = my_mws.modeler

    idcnt = 0
    id_reset  = 0
    time_loop_start = time.time()
    for i in range(0, 3000):
        change_para(my_modeler, "a", ','.join(str(j) for j in a[i]))
        change_para(my_modeler, "h", ','.join(str(j) for j in h[i]))
        change_para(my_modeler, "e", ','.join(str(j) for j in e[i]))
        my_modeler.full_history_rebuild()
        my_modeler.run_solver()

        my_project = cst.results.ProjectFile()
        my_project.init(my_project_path, True)
        s11 = my_project.get_3d().get_result_item(r"1D Results\S-Parameters\S1,1")
        num = len(s11)
        s11_data = np.array(s11.get_data())
        s11_freq = np.array(np.abs(s11_data[:, 0]), dtype=float).reshape([num, 1])
        s11_s11 = np.array(s11_data[:, 1], dtype=complex).reshape([num, 1])
        s11_mag = np.abs(s11_s11).reshape([num, 1])
        s11_phase = np.angle(s11_s11, deg=True).reshape([num, 1])

        df_name = ["freq", "mag", "phase"]
        df_data = np.zeros((num, 3))
        for df_i in range(num):
            df_data[df_i, 0] = s11_freq[df_i]
            df_data[df_i, 1] = s11_mag[df_i]
            df_data[df_i, 2] = s11_phase[df_i]

        df = pd.DataFrame(columns=df_name, data=df_data)
        df.to_csv(f'../data/s11/s11_{i}.csv', encoding='utf-8', index=False)

        time_loop_end = time.time()
        if time_loop_end - time_loop_start >= 600:
            my_mws.save()
            my_mws.close()
            my_de.close()
            id_reset += 1
            time.sleep(5)
            time_loop_start = time.time()
            my_de = cst.interface.DesignEnvironment()
            my_mws = my_de.open_project(my_project_path)
            my_modeler = my_mws.modeler

        time_end = time.time()
        print(
            f'No.{idcnt}, ID:{ids[i]}, RunTime:{time_end - time_start}, AvgTime:{(time_end - time_start) / (idcnt + 1)}')
        idcnt += 1
    my_mws.save()
