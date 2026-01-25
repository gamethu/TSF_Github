from fastapi import FastAPI
from routes.stationRoute import router as stationRoute
from routes.modelRoute import router as modelRoute

app = FastAPI()

app.include_router(stationRoute)
app.include_router(modelRoute)

@app.get("/")
def read_root():
    return {"status"      : "ok",
            "description" : "API are working..."}