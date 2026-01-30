from pathlib import Path
from fastapi import HTTPException
from modules.Model import Repository


async def fetch_data(type="all"):
    data = await Repository.getAll()
    
    if type == "all"          : return data
    if type == "name"         : return list(data.keys())
    # if type == "measure_unit" : return dict({'YEAR'       : "", 
    #                                          'MONTH'      : "", 
    #                                          'DAY'        : "",
    #                                          'Nina_index' : "°C",
    #                                          'DEW_ave'    : "°C",
    #                                          'TEMP_ave'   : "°C",
    #                                          'TEMP_max'   : "°C",
    #                                          'RH_ave'     : "%",
    #                                          'DEW_max'    : "°C",
    #                                          'RH_max'     : "%",
    #                                          'sp_ave'     : "kPa",
    #                                          'tcc_ave'    : "%",
    #                                          'tp_sum'     : "mm",
    #                                          'ws_ave'     : "m/s",
    #                                          'wd_ave'     : "°"})