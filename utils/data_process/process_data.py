import numpy as np
from openpyxl import load_workbook
from torch.utils import data


def normalize(data):
    """Normalize data, input type list"""
    data = np.array(data)
    max_val = data.max()
    min_val = data.min()
    ret = (data - min_val) / (max_val - min_val)
    return list(ret)


def get_sheet_col(sheet, idx):
    """Acitve sheet and column idx"""
    ret = [cell.value for cell in list(sheet.columns)[idx]]
    return ret


def get_sheet_row(sheet, idx):
    """Acitve sheet and row idx"""
    ret = [cell.value for cell in list(sheet.rows)[idx]]
    return ret
