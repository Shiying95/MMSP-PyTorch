import json
import pandas as pd

def write_new_attr_cols(arrows_root):
    with open(f'{arrows_root}/label_encoding.json', 'r') as fp:
        le = json.load(fp)
    
    info = dict()
    info['cate_cols'] = {}

    for col, values in le.items():
        info['cate_cols'].update({col: len(values)})

    info['non_cate_cols'] = ['jd_prc'] 
    info['info_cols'] = ['item_sku_id', 'item_id']
    info['label_cols'] = ['sale_qtty']
    info["region_cols"] = ["dummy"]
    info["dt_cols"] = ["on_shelf_periods"]
    # info["img_cols"] = ["img_feature"]
    # info["text_cols"] = ["text_feature"]

    for col in ['style', '适用季节']:
        if col in info['cate_cols']:
            info['tit_cols'] = {
                col: len(le[col])        
                }

    info['cate_cols'].update({
        "year_no": 2022,
        "week_no": 54, 
        "month_no": 13, 
        "day_no": 32,
        })
    
    with open(f'{arrows_root}/new_attr_cols.json', 'w') as fp:
        json.dump(info, fp)
        
    print('write attr cols file successfully!')