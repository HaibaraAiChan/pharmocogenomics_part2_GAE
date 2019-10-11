
import numpy as np
import os
import pandas as pd


def check_duplicated(df, file_path):
    df_change = pd.read_csv('./first80.csv', delimiter=',')

    # df['Protein_name'] = df['Protein_name'].map(df40.set_index('Protein')['ISO_Protein'])
    df_uni = df_change.set_index('Protein')
    key_dict = df_uni.to_dict()

    df.loc[:, 'Protein_name'] = df['Protein_name'].replace(key_dict['ISO_Protein'])
    # print(df[df['Protein_name'] == 'ENSP00000360423'])
    df = df[['Protein', 'Protein_name']]
    df.to_csv(file_path, index=False)


    return


if __name__ == "__main__":
    file = '../../data/positive-uni-ENSP'
    df = pd.read_csv(file, delimiter=',')
    check_duplicated(df, file)