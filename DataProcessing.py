
import pandas as pd

'''
    Returns: 
        label_data: nx(hxw) flattened array, containing patches corresponding to head/tail
        input_img_data: nx(hxw) flattened array, containing original frame
        tail_position_data: nx2
        head_position_data: nx2
'''
def collectAllData_v2(path):
    df = pd.read_csv(path)
    df = df.set_index('Filename')
    return df