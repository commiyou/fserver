import datetime
import shutil
from pathlib import Path
from typing import Optional

import human_readable
import pandas as pd
from cachetools import TTLCache, cached
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse, HTMLResponse, JSONResponse

cache = TTLCache(maxsize=30, ttl=3600 * 48)  # 缓存最多10个文件，每个文件缓存48h


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/list/{file_path:path}")
async def list_files(request: Request, file_path: str):
    path = Path(file_path)
    if path.is_file():
        return {"filename": path.name}
    else:
        files = [
            {
                "name": x.name,
                "type": "dir" if x.is_dir() else "file",
                "size": x.stat().st_size,
                "human_size": human_readable.file_size(x.stat().st_size, gnu=True),
                "time": x.stat().st_mtime,
                "human_time": human_readable.date_time(
                    datetime.datetime.fromtimestamp(x.stat().st_mtime)
                ),
            }
            for x in path.iterdir()
        ]

        return templates.TemplateResponse(
            "list.html", {"request": request, "files": files, "path": file_path}
        )


@app.post("/upload/{file_path:path}")
async def upload_file(file_path: str, file: UploadFile = File(...)):
    path = Path(file_path) / file.filename

    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": path.name}


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    return FileResponse(file_path)


@cached(cache)
def read_file(file_path: str):
    path = Path(file_path)
    if path.is_file():
        df = pd.read_csv(path, sep="\t")
        # fix DataTables warning: table id=tsvTable - Requested unknown parameter '优化后的sug-test_299100_10w_10w.gsb.price' for row 0, column 3.
        df.columns = map(lambda x: x.replace(".", "-"), df.columns)
        return df
    else:
        return None


@app.get("/tsv/{file_path:path}")
async def read_tsv(
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 1000,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
):
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and file_path in cache:
        del cache[file_path]
    df = read_file(file_path)
    if df is None:
        return {"error": "File not found."}
    columns = df.columns.tolist()
    name = path.name
    return templates.TemplateResponse(
        "tsv.html",
        {
            "request": request,
            "path": file_path,
            "name": name,
            "columns": columns,
            "length": length,
            "start": start,
        },
    )


@app.get("/api/tsv/key/{file_path:path}")
async def api_tsv_key(
    request: Request,
    file_path: str,
    key: str,
    reload: Optional[bool] = False,
):
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and file_path in cache:
        del cache[file_path]

    df = read_file(file_path)
    keys = df[key].dropna().unique().tolist()

    return JSONResponse(keys)


@app.get("/api/tsv/{file_path:path}")
async def api_tsv(
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 100,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
    draw: Optional[int] = -1,
):
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and file_path in cache:
        del cache[file_path]

    df = read_file(file_path)

    if df is None:
        return {"error": "File not found."}

    if key and value:
        df = df[df[key] == value]

    # 获取搜索参数
    search_value = request.query_params.get("search[value]", "")
    if search_value:
        filtered_df = df[
            df.apply(
                lambda row: row.astype(str).str.contains(search_value).any(), axis=1
            )
        ]
    else:
        filtered_df = df

    # # 获取排序参数
    # order_column = request.query_params.get("order[0][column]", "0")
    # order_dir = request.query_params.get("order[0][dir]", "asc")
    # filtered_df.sort_values(
    #     by=filtered_df.columns[int(order_column)],
    #     ascending=(order_dir == "asc"),
    #     inplace=True,
    # )
    print(
        len(filtered_df),
        start,
        length,
        filtered_df[start : start + 1].to_dict(orient="records"),
    )

    return JSONResponse(
        {
            "data": filtered_df[start : start + length]
            .fillna("")
            .to_dict(orient="records"),
            "recordsTotal": len(df),
            "recordsFiltered": len(filtered_df),
            "draw": draw,
        }
    )
