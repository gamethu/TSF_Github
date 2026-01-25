from pathlib import Path
from fastapi import HTTPException
from modules.Model import Repository


async def fetch_data(type="all"):
    data = await Repository.getAll()
    
    if type == "all"  : return data
    if type == "name" : return list(data.keys())