import datetime
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import aiofiles
import human_readable
import pandas as pd
from cachetools import TTLCache
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from pandas import DataFrame, ExcelWriter
from starlette.responses import FileResponse, JSONResponse, RedirectResponse

cache = TTLCache(maxsize=30, ttl=3600 * 48)  # 缓存最多10个文件，每个文件缓存48h


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def get_directory_contents(directory: Path | str) -> list[dict[str, Any]]:
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
        {
            "request": request,
            "files": files,
            "path": file_path,
            "breadcrumbs": breadcrumbs,
        },
    )


@app.post("/upload/{file_path:path}")
async def upload_file(file_path: str, file: UploadFile = File(...)) -> RedirectResponse:  # noqa: B008
    """upload file to `file_path`"""
    if not file:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")
    assert file.filename is not None
    path = Path(file_path) / file.filename
    i = 1
    while path.exists():
        path = Path(file_path) / f"{file.filename}.new{i}"
        i += 1

    # with path.open("wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    async with aiofiles.open(path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    url = app.url_path_for("list", file_path=file_path)
    response = RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    return response


@app.get("/excel/{file_path:path}")
async def download_excel(
    file_path: str,
    names: Optional[str] = None,
    header: Optional[bool] = True,
    json_cols: Optional[str] = None,
):
    """download file as excel"""
    path = Path(file_path)

    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if path.suffix in {".xls", ".xlsx"}:
        return FileResponse(path)
    k = (file_path,)
    if k in cache:
        del cache[k]
    names_list = None
    if names:
        names_list = [str(x) for x in names.split(",")]
    df = read_file(file_path, names_list, header=header)

    if df is None:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        if path.suffix in [".tsv", ".csv", ".data", "txt"]:
            excel_path = path.with_suffix(".xlsx")
        else:
            excel_path = Path(str(path) + ".xlsx")

        excel_file_name = excel_path.stem + ".xlsx"
        with ExcelWriter(excel_path) as writer:
            df.to_excel(writer, index=False)
        return FileResponse(excel_path, filename=excel_file_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/download/{file_path:path}")
async def download_file(file_path: str) -> Response:
    """download file"""
    file_path = file_path.strip()
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if os.path.isdir(file_path):
        raise HTTPException(status_code=400, detail=f"Path is a directory: {file_path}.")
    return FileResponse(file_path)


def to_img_tag(x: Any) -> str:
    """convert url to img tag"""
    if pd.isna(x) or not isinstance(x, str) or not x.strip():
        return ""
    return f'<img src="{x}" height="100" />'


def read_file(
    file_path: str,
    names_list: tuple[str, ...] | None = None,
    *,
    header: bool = True,
) -> DataFrame | None:
    """以file name为key load&cache文件"""
    k = (file_path,)
    if k in cache:
        return cache[k]
    path = Path(file_path)
    print(header)
    if path.stat().st_size == 0:
        return None
    if path.is_file():
        print(path.suffix, file=sys.stderr)
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        elif path.suffix in {".json", ".ndjson"}:
            df = pd.read_json(path, dtype=str, lines=True)
        else:
            try:
                if names_list:
                    df = pd.read_csv(path, sep="\t", names=list(names_list))
                    df.columns = [str(x).replace(".", "-") for x in df.columns]
                elif header:
                    df = pd.read_csv(path, sep="\t")
                else:
                    df = pd.read_csv(path, sep="\t", header=None)
                    df.columns = [f"col{i}" for i in range(df.shape[1])]
            except pd.errors.ParserError:
                # Fallback for ragged files (variable column counts)
                # "Good Taste": Don't crash, handle the data structure as it is (a ragged matrix)
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        lines = [line.rstrip("\n").split("\t") for line in f]
                    
                    if not lines:
                        return None
                        
                    df = pd.DataFrame(lines)
                    
                    if names_list:
                        # Best effort: Assign provided names to the first N columns
                        # If data has more columns than names, they keep default int indices or we can rename them
                        current_cols = list(df.columns)
                        limit = min(len(names_list), len(current_cols))
                        
                        new_cols = list(names_list)[:limit] + [f"col{i}" for i in range(limit, len(current_cols))]
                        df.columns = new_cols
                        df.columns = [str(x).replace(".", "-") for x in df.columns]
                        
                    elif header:
                        # First row is header
                        if len(df) > 0:
                            # Extract headers from the first row of the data (which is now in the DF)
                            # Note: lines[0] might be shorter than max width of DF
                            headers = lines[0]
                            # Pad headers if the DF is wider than the first row
                            if len(df.columns) > len(headers):
                                headers += [f"col{i}" for i in range(len(headers), len(df.columns))]
                            
                            df.columns = headers
                            # Drop the first row as it's now the header
                            df = df.iloc[1:].reset_index(drop=True)
                    else:
                        # No header, just rename all columns
                        df.columns = [f"col{i}" for i in range(df.shape[1])]
                    
                    # Attach a warning to the DataFrame to notify the UI
                    df.attrs["warning_msg"] = "⚠️ 注意：检测到文件列数不规则（Ragged Data）。已启用兼容模式加载，部分列可能自动填充为空值。"

                except Exception as e:
                     # If fallback also fails, then we truly fail
                    raise HTTPException(status_code=400, detail=f"CSV/TSV Parse Error: {e}") from e

        cache[k] = df
        return df
    return None


def get_col_indices(df: DataFrame, cols: Optional[str]) -> list[int]:
    """Resolve column names or indices to a list of 0-based integer indices."""
    if not cols:
        return []
    indices = []
    for x in cols.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            # Try as integer index
            idx = int(x)
            if 0 <= idx < len(df.columns):
                indices.append(idx)
        except ValueError:
            # Try as column name
            if x in df.columns:
                indices.append(df.columns.get_loc(x))
    return indices


@app.get("/tsv/{file_path:path}", name="tsv")
async def read_tsv(  # noqa: PLR0917
    request: Request,
    file_path: str,
    start: Optional[int] = 0,
    length: Optional[int] = 1000,
    reload: Optional[bool] = False,
    key: Optional[str] = None,
    value: Optional[str] = None,
    names: Optional[str] = None,
    header: Optional[bool] = True,
    json_link_cols: Optional[str] = None,
    json_cols: Optional[str] = None,
    image_cols: Optional[str] = None,
    hide_cols: Optional[str] = None,
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

    df = read_file(
        file_path,
        names_list,
        header=header,
    )
    if df is None:
        return {"error": "File not found or empty."}

    # Resolve columns to indices
    json_link_col_list = get_col_indices(df, json_link_cols)
    json_col_list = get_col_indices(df, json_cols)
    image_col_list = get_col_indices(df, image_cols)
    hide_col_list = get_col_indices(df, hide_cols)

    columns = df.columns.tolist()
    name = path.name
    parent_path = path.parent
    
    # Check if a warning was attached during parsing (e.g. fallback triggered)
    warning_msg = getattr(df, "attrs", {}).get("warning_msg", None)

    return templates.TemplateResponse(
        "tsv.html",
        {
            "request": request,
            "path": file_path,
            "parent_path": str(parent_path),
            "name": name,
            "columns": columns,
            "length": length,
            "start": start,
            "recordsTotal": len(df),
            "json_link_cols": json_link_col_list,  # Pass json_link_cols to the template
            "json_cols": json_col_list,  # Pass json_cols to the template
            "image_cols": image_col_list,
            "hide_cols": hide_col_list,
            "warning_msg": warning_msg,
        },
    )


@app.get("/txt/{file_path:path}")
async def get_txt_content(file_path: str) -> PlainTextResponse:
    """
    返回指定路径的txt文件内容作为纯文本。

    - **file_path**: 文件的相对路径，可以包含子目录。
    """
    path = Path(file_path)

    # 检查文件是否存在且为txt文件
    if not path.is_file():
        raise HTTPException(status_code=404, detail="TXT文件未找到")

    try:
        async with aiofiles.open(path, encoding="utf-8") as f:
            content = await f.read()
        return PlainTextResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {e}")


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
        del cache[file_path,]

    df = read_file(file_path)
    assert df is not None
    keys = df[str(key)].dropna().unique().tolist()

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
    json_link_cols: Optional[str] = None,
    json_cols: Optional[str] = None,
):
    """return table data as json for datatables ajax call"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    if reload and (file_path,) in cache:
        del cache[file_path,]

    # Pass json_col_list as json_cols to apply pretty-printing
    # Pass json_link_col_list as raw_json_cols to prevent pretty-printing for clickable columns
    df = read_file(file_path)

    if df is None:
        return {"error": "File not found."}

    records_total = len(df)

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
        df_filtered = df[df[key] == converted_value]
    else:
        df_filtered = df

    # 获取搜索参数
    search_value = request.query_params.get("search[value]", "")
    if search_value:
        filtered_df = df_filtered[
            df_filtered.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)
        ]
    else:
        filtered_df = df_filtered

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
            "recordsTotal": records_total,
            "recordsFiltered": len(filtered_df),
            "draw": draw,
        },
    )


@app.get("/api/tsv/cell/{file_path:path}")
async def get_tsv_cell_content(
    file_path: str,
    row_index: int,
    col_index: int,
    names: Optional[str] = None,
    header: Optional[bool] = True,
) -> JSONResponse:
    """Retrieve raw content of a specific cell in a TSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    names_list = tuple(names.split(",")) if names else None

    # Read the file with no json_cols processed to get raw content
    df = read_file(file_path, names_list, header=header)  # Ensure raw content

    if df is None:
        raise HTTPException(status_code=404, detail="File not found or empty.")

    try:
        cell_value = df.iloc[row_index, col_index]
        return JSONResponse({"content": str(cell_value)})
    except IndexError:
        raise HTTPException(status_code=400, detail="Row or column index out of bounds.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cell content: {e}")


def get_db_connection(db_path: str):
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Database path '{db_path}' does not exist.")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 以字典形式返回行
        return conn
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/db/{db_path:path}")
def read_db(
    db_path: Path,
    table: Optional[str] = Query(None, description="Name of the table to query."),
    col: Optional[str] = Query(None, description="Column name for filtering."),
    value: Optional[str] = Query(None, description="Value for filtering."),
    limit: int = Query(10, ge=1, description="Limit the number of returned records."),
):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if table:
        # 查询指定表的数据
        # try:
        #     cursor.execute(f"SELECT * FROM {table}")
        #     rows = cursor.fetchmany(size=10)
        #     # 获取列名
        #     columns = rows[0].keys() if rows else []
        #     data = [dict(row) for row in rows]
        #     return {"table": table, "columns": columns, "data": data}
        # except sqlite3.Error as e:
        #     raise HTTPException(status_code=400, detail=f"Error querying table '{table}': {str(e)}")
        # finally:
        #     conn.close()
        # 验证表名是否存在，防止 SQL 注入
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(
                status_code=400,
                detail=f"Table '{table}' does not exist in the database.",
            )

        # 构建查询语句
        base_query = f"SELECT * FROM {table}"
        parameters = []

        if col and value:
            # 验证列名是否存在
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row["name"] for row in cursor.fetchall()]
            if col not in columns:
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' does not exist in table '{table}'.",
                )
            base_query += f" WHERE {col} = ?"
            parameters.append(value)

        base_query += " LIMIT ?"
        parameters.append(limit)

        try:
            cursor.execute(base_query, parameters)
            rows = cursor.fetchall()
            # 获取列名
            columns = rows[0].keys() if rows else []
            data = [dict(row) for row in rows]
            return {"table": table, "columns": columns, "data": data}
        except sqlite3.Error as e:
            raise HTTPException(status_code=400, detail=f"Error querying table '{table}': {e!s}") from e
        finally:
            conn.close()
    else:
        # 获取数据库中所有表及其模式
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row["name"] for row in cursor.fetchall()]
            db_info = {}
            for tbl in tables:
                cursor.execute(f"PRAGMA table_info({tbl});")
                columns = [dict(col) for col in cursor.fetchall()]
                db_info[tbl] = columns
            return {"tables": db_info}
        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            conn.close()


@app.get("/json_viewer")
async def json_viewer(content: str = Query(..., description="JSON content to display")):
    """Displays JSON content in a readable format."""
    try:
        parsed_json = json.loads(content)
        pretty_json = json.dumps(parsed_json, indent=4, ensure_ascii=False)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>JSON Viewer</title>
    <style>
        body {{ background-color: #f0f0f0; margin: 20px; }}
        pre {{ background-color: #ffffff; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <pre><code>{pretty_json}</code></pre>
</body>
</html>"""
        return Response(content=html_content, media_type="text/html")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON content")


if __name__ == "__main__":
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s " + LOGGING_CONFIG["formatters"]["access"]["fmt"]
    uvicorn.run("fserver:app", host="0.0.0.0", port=8113, reload=True)
