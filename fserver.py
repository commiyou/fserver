import datetime
import os
import shutil
from pathlib import Path
from typing import Optional

import human_readable
import pandas as pd
from cachetools import TTLCache
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.templating import Jinja2Templates
from pandas import ExcelWriter
from starlette.responses import FileResponse, JSONResponse, RedirectResponse

cache = TTLCache(maxsize=30, ttl=3600 * 48)  # 缓存最多10个文件，每个文件缓存48h


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def get_directory_contents(directory: Path) -> list[dict]:
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
                "human_time": human_readable.date_time(datetime.datetime.fromtimestamp(item.stat().st_mtime)),  # noqa: DTZ006
            }
            contents.append(item_info)
    # contents.sort(lambda x: x["time"], reverse=True)
    return contents


@app.get("/list/{file_path:path}", name="list")
async def list_files(request: Request, file_path: Path) -> Response:
    """list files in `file_path`"""
    path = Path(file_path)
    if path.is_file():
        url = app.url_path_for("tsv", file_path=file_path)
        response = RedirectResponse(url=url)
        return response
    files = get_directory_contents(file_path)
    breadcrumbs = [{"name": part, "url": "/" + "/".join(path.parts[: i + 1])} for i, part in enumerate(path.parts)]

    return templates.TemplateResponse(
        "list.html",
        {"request": request, "files": files, "path": file_path, "breadcrumbs": breadcrumbs},
    )


@app.post("/upload/{file_path:path}")
async def upload_file(file_path: str, file: UploadFile = File(...)):  # noqa: ANN201
    """upload file to `file_path`"""
    if not file:
        return {"error": "no file upload"}
    assert file.filename is not None
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


@app.get("/excel/{file_path:path}")
async def download_excel(file_path: str):  # noqa: ANN201
    """download file as excel"""
    path = Path(file_path)

    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if path.suffix in [".xls", ".xlsx"]:
        return FileResponse(path)
    k = (file_path,)
    if k in cache:
        del cache[k]
    df = read_file(file_path)  # noqa: PD901

    if df is None:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        if path.suffix in [".tsv", ".csv", ".data", "txt"]:
            excel_path = path.with_suffix(".xlsx")
        else:
            excel_path = Path(str(path) + ".xlsx")

        excel_file_name = path.stem + ".xlsx"
        with ExcelWriter(excel_path) as writer:
            df.to_excel(writer)
        return FileResponse(excel_path, filename=excel_file_name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/download/{file_path:path}")
async def download_file(file_path: str) -> Response:
    """download file"""
    return FileResponse(file_path)


def read_file(file_path: str, names_list: tuple[str, ...] | None = None):  # noqa: ANN201
    """以file name为key load&cache文件"""
    k = (file_path,)
    if k in cache:
        return cache[k]
    path = Path(file_path)
    if path.is_file():
        df = pd.read_csv(path, sep="\t", names=list(names_list) if names_list else None)
        # fix DataTables warning:
        # table Requested unknown parameter '优化后的sug-test_299100_10w_10w.gsb.price' for row 0, column 3.
        df.columns = [str(x).replace(".", "-") for x in df.columns]
        cache[k] = df
        return df
    return None


@app.get("/tsv/{file_path:path}", name="tsv")
async def read_tsv(  # noqa: ANN201
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 1000,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
    names: Optional[str] = None,
):
    """show tabluar page of tsv file using pandas display"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    k = (file_path,)
    print(f"reload {reload}, incache {k in cache}, {file_path}")
    if reload and k in cache:
        del cache[k]
    names_list = None
    if names:
        names_list = tuple(names.split(","))

    df = read_file(file_path, names_list)  # noqa: PD901
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
async def api_tsv_key(  # noqa: ANN201
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
    assert df is not None
    keys = df[key].dropna().unique().tolist()

    return JSONResponse(keys)


@app.get("/api/tsv/{file_path:path}")
async def api_tsv(
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 500,
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

    df = read_file(file_path)  # noqa: PD901

    if df is None:
        return {"error": "File not found."}

    # print("columns", df.columns, "key", key, f"value:{type(value)}", value, "@@")
    if key and value is not None:
        column_dtype = df[key].dtypes
        print("Column data type:", column_dtype)

        # Convert the object `s` to the column's data type
        if column_dtype == "int64":
            converted_value = int(value)
        elif column_dtype == "float64":
            converted_value = float(value)
        elif column_dtype == "bool":
            converted_value = bool(value)
        elif column_dtype == "datetime64[ns]":
            converted_value = pd.to_datetime(value)
        else:
            converted_value = str(value)
        df = df[df[key] == converted_value]

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
        },
    )
