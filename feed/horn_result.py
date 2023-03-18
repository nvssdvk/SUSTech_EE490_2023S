import datetime
import os
import time
import numpy as np
import pandas as pd

import cst.results

line_break = '\n'

if __name__ == '__main__':
    # my_project_path = os.path.abspath(r"D:\User\Course\EE490.2023S\src\data\RA_23_03_09\cell.cst")
    my_project_path = r"D:\User\Course\EE490.2023S\SUSTech_EE490_2023S\cst\horn.cst"

    my_project = cst.results.ProjectFile(my_project_path)
    tree_items = my_project.get_3d().get_tree_items()
    tree_items = line_break.join(tree_items)
    print(tree_items)
    # df = pd.DataFrame(columns=df_name, data=df_data)
    # df.to_csv(f'../data/s11/s11_{i}.csv', encoding='utf-8', index=False)
