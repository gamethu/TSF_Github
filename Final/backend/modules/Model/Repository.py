from pathlib import Path
from fastapi import HTTPException

async def getAll():
    base_dir = Path(__file__).resolve().parent
    return dict({"CaMau" : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/CaMau_90.24_cleaned.csv").resolve(),
                 "DH"    : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/DH_90.24_cleaned.csv").resolve(),
                 "NB"    : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/CaMau_90.24_cleaned.csv").resolve(),
                 "QN"    : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/QN_90.24_cleaned.csv").resolve(),
                 "TH"    : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/TH_90.24_cleaned.csv").resolve(),
                 "TSN"   : (base_dir / "../../../../Drafts/Temp Prediction/data/processed/datasets/TSN_90.24_cleaned.csv").resolve()})
async def findByCountryName(name):
    pass
async def findByStoreID(id):
    pass
async def findAll():
    pass