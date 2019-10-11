import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
    input_folder = '../data/index_ENSP/target_cell_line/'
    input_cell_line = '../data/index_ENSP/cellline_index.csv'
    folder_files = os.listdir(input_folder)
    files = [i[:-4]  for i in folder_files]
    # folder_files = folder_files[:, 0:-4]
    cellline_data = pd.read_csv(input_cell_line, delimiter=',')
    # cellline_data['comb'] = cellline_data['Protein'].map(str)+'_'+cellline_data['Drug'].map(str)+'_'+cellline_data['Cell_lines'].map(str)
    cellline_data['comb'] = cellline_data['Protein'].map(str) + '_' + cellline_data['Drug'].map(str) + '_' + \
                            cellline_data['Cell_lines'].map(str)
    res = pd.unique(cellline_data[['comb']].values.ravel('K'))
    # rest[['Protein','Drug','Cell_lines']]
    length = len(res)
    res = [res[i].split('_') for i in range(length)]
    res = np.array(res)
    df_res = pd.DataFrame({'Protein':res[:,0],'Drug':res[:,1],'Cell_lines':res[:,2]})
    df_res.to_csv(input_cell_line, index=False)

    du = cellline_data[cellline_data.duplicated()]
    du.to_csv('duplicated_cell_line_target.csv', index=False)
    print('-'*40)
    folder_files
    cellline_data
    print('pre-process data-set is ready')