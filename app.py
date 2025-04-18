from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathfinder import solve_task, load_task_waypoints
import traceback
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
  CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
  allow_headers=["*"], allow_credentials=True
)


@app.get("/route")
def get_route():
    try:
        path  = solve_task()
        tasks = load_task_waypoints()
        return {"path": path, "tasks": tasks}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)   # logs the full traceback to your console
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb.splitlines()}
        )


    
 
@app.get("/ping")
def ping():
    return {"ping": "pong"}
