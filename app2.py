from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathfinder import solve_task

app = FastAPI()

# Enable CORS so React (http://localhost:5173) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/route")
async def get_route():
    """
    Compute and return the path as a list of [lon, lat] pairs.
    """
    coords = solve_task()
    return {"path": coords}

# To run:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
