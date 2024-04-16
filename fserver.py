import datetime
import os
import shutil
from pathlib import Path
from typing import Optional

import human_readable
import pandas as pd
from cachetools import TTLCache, cached
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.templating import Jinja2Templates
from starlette.responses import (FileResponse, HTMLResponse, JSONResponse,
                                 RedirectResponse)

cache = TTLCache(maxsize=30, ttl=3600 * 48)  # 缓存最多10个文件，每个文件缓存48h


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def get_directory_contents(directory):
    """return dir content"""
    contents = []
    for item in os.scandir(directory):
        if os.path.exists(item):
            item_info = {
                "name": item.name,
                "size": item.stat().st_size if item.is_file() else "",
                "time": item.stat().st_mtime,
                "type": "file" if item.is_file() else "dir",
                "human_size": human_readable.file_size(item.stat().st_size, gnu=True),
                "human_time": human_readable.date_time(datetime.datetime.fromtimestamp(item.stat().st_mtime)),
            }
            contents.append(item_info)
    contents.sort(lambda x: x["time"], reverse=True)
    return contents


@app.get("/list/{file_path:path}", name="list")
async def list_files(request: Request, file_path: str):
    """list files in `file_path`"""
    path = Path(file_path)
    if path.is_file():
        url = app.url_path_for("tsv", file_path=file_path)
        response = RedirectResponse(url=url)
        return response
    else:
        files = get_directory_contents(file_path)
        breadcrumbs = [{"name": part, "url": "/" + "/".join(path.parts[: i + 1])} for i, part in enumerate(path.parts)]

        return templates.TemplateResponse(
            "list.html", {"request": request, "files": files, "path": file_path, "breadcrumbs": breadcrumbs}
        )


@app.post("/upload/{file_path:path}")
async def upload_file(file_path: str, file: UploadFile = File(...)):
    """upload file to `file_path`"""
    if not file:
        return {"error", "no file upload"}
    path = Path(file_path) / file.filename
    i = 1
    while path.exists():
        path = Path(file_path) / f"{file.filename}.new{i}"
        i += 1

    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    url = app.url_path_for("list", file_path=file_path)
    response = RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    return response


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """download file"""
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


@app.get("/tsv/{file_path:path}", name="tsv")
async def read_tsv(
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 1000,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
):
    """show tabluar page of tsv file using pandas display"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    print(f"reload {reload}, incache {file_path in cache}, {file_path}, {cache}")
    if reload and (file_path,) in cache:
        del cache[(file_path,)]
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
            "recordsTotal": len(df),
        },
    )


@app.get("/api/tsv/key/{file_path:path}")
async def api_tsv_key(
    request: Request,
    file_path: str,
    key: str,
    reload: Optional[bool] = False,
):
    """tsv html get keys of tsv file"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and (file_path,) in cache:
        del cache[(file_path,)]

    df = read_file(file_path)
    keys = df[key].dropna().unique().tolist()

    return JSONResponse(keys)


@app.get("/api/tsv/{file_path:path}")
async def api_tsv(
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 1000,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
    draw: Optional[int] = -1,
):
    """tsv html get content of tsv file"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and (file_path,) in cache:
        del cache[(file_path,)]

    df = read_file(file_path)

    if df is None:
        return {"error": "File not found."}

    if value is not None and value.isdigit():
        value = int(value)
    if key and value is not None:
        df = df[df[key] == value]

    # 获取搜索参数
    search_value = request.query_params.get("search[value]", "")
    if search_value:
        filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)]
    else:
        filtered_df = df

    # print(
    #         len(filtered_df),
    #         start,
    #         length,
    #         filtered_df[start : start + 1].to_dict(orient="records"),
    #         )

    return JSONResponse(
        {
            "data": (filtered_df[start : start + length] if length > 0 else filtered_df[start:])
            .fillna("")
            .to_dict(orient="records"),
            "recordsTotal": len(df),
            "recordsFiltered": len(filtered_df),
            "draw": draw,
        }
    )
