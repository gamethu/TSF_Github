from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List
from modules.Model import Controller

router = APIRouter(tags=["Station"])

@router.get("/stations/{type}")
async def get_country_data(type): return await Controller.get(type)