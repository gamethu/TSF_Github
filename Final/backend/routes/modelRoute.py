from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List
from modules.Model import Controller

router = APIRouter(tags=["Models"])

@router.get("/models/{type}")
async def get_model_data(type): return await Controller.get(type)