from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List
from modules.Station import Controller

router = APIRouter(tags=["Stations"])

@router.get("/stations/{type}")
async def get_station_data(type): return await Controller.get(type)