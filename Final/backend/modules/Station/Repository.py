from pathlib import Path
from fastapi import HTTPException


async def getAll():
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "../../../../../Drafts/data/processed").resolve()
    
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")
    
    res = {}
    for country in data_dir.iterdir():
        res[country.name] = dict()
        for storeid in country.iterdir():
            res[country.name][storeid.name.replace(".csv","")] = str(storeid.resolve())
    return res
async def findByCountryName(name):
    pass
async def findByStoreID(id):
    pass
async def findAll():
    pass