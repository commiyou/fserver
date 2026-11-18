import datetime
import html as html_mod
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Optional

import aiofiles
import human_readable
import markdown as md_lib
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

# Excel 不允许的控制字符（TAB/LF/CR 除外）
_EXCEL_ILLEGAL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Excel 临时文件目录
EXCEL_TEMP_DIR = Path(tempfile.gettempdir()) / "fserver_excel"
EXCEL_TEMP_DIR.mkdir(exist_ok=True)


def _clean_for_excel(df: DataFrame) -> DataFrame:
    """移除 DataFrame 中 Excel 不支持的控制字符。"""
    def _clean(val: Any) -> Any:
        if isinstance(val, str):
            return _EXCEL_ILLEGAL_CHARS_RE.sub("", val)
        return val
    return df.map(_clean)


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def _quote_route_path(path: Path) -> str:
    """Percent-encode path components while preserving filesystem separators."""
    from urllib.parse import quote

    absolute = bool(path.parts and path.parts[0].startswith("/"))
    parts = [
        quote(part, safe="")
        for part in path.parts
        if part and not part.startswith("/") and part != "."
    ]
    encoded = "/".join(parts)
    return f"/{encoded}" if absolute else encoded


def _route_url(prefix: str, path: Path) -> str:
    """Build a route URL without encoding filesystem separators."""
    return f"{prefix}/{_quote_route_path(path)}"


def _file_open_url(path: Path) -> str:
    """Return the default action URL for a file."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _route_url("/md", path)
    if suffix == ".tsv":
        return _route_url("/tsv", path)
    if suffix == ".csv":
        return f"{_route_url('/tsv', path)}?delim=%2C"
    return _route_url("/download", path)


def _file_preview_url(path: Path) -> str | None:
    """Return a preview URL only for document formats with a dedicated viewer."""
    if path.suffix.lower() in {".md", ".tsv", ".csv"}:
        return _file_open_url(path)
    return None


def _canonical_route_redirect(request: Request, prefix: str, path: Path) -> RedirectResponse | None:
    """Redirect legacy percent-encoded navigation URLs to their clean form."""
    raw_path = request.scope.get("raw_path", b"").decode("ascii", errors="ignore")
    canonical = _route_url(prefix, path)
    if raw_path == canonical:
        return None
    query = request.url.query
    if query:
        canonical = f"{canonical}?{query}"
    return RedirectResponse(canonical, status_code=status.HTTP_308_PERMANENT_REDIRECT)


def get_directory_contents(directory: Path | str) -> list[dict[str, Any]]:
    """Return directory entries with metadata and ready-to-use URLs."""
    contents = []
    base_path = Path(directory)
    for item in os.scandir(base_path):
        try:
            stat_result = item.stat()
        except OSError:
            continue
        if os.path.exists(item.path):
            is_dir = item.is_dir()
            item_path = base_path / item.name
            mtime = stat_result.st_mtime
            item_info = {
                "name": item.name,
                "size": stat_result.st_size if not is_dir else None,
                "time": mtime,
                "type": "dir" if is_dir else "file",
                "human_size": human_readable.file_size(stat_result.st_size, gnu=True) if not is_dir else "—",
                "human_time": human_readable.date_time(datetime.datetime.fromtimestamp(mtime)),  # noqa: DTZ006
                "date_str": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),  # noqa: DTZ006
                "url": _route_url("/list", item_path),
            }
            if not is_dir:
                item_info["url"] = _file_open_url(item_path)
                preview_url = _file_preview_url(item_path)
                if preview_url:
                    item_info["preview_url"] = preview_url
                item_info["download_url"] = _route_url("/download", item_path)
            contents.append(item_info)
    return contents


def get_breadcrumbs(path: Path) -> list[dict[str, str]]:
    """Return breadcrumb entries whose URLs are safe to use directly."""
    breadcrumbs: list[dict[str, str]] = []
    for index, part in enumerate(path.parts):
        if part in {"", ".", "/"}:
            continue
        target = Path(*path.parts[: index + 1])
        breadcrumbs.append({"name": part, "url": _route_url("/list", target)})
    return breadcrumbs


def render_breadcrumbs_html(parent_path: Path, current_name: str | None) -> str:
    """Render the shared breadcrumb component used by list and Markdown pages."""
    parts = [
        '<ol class="breadcrumbs" aria-label="面包屑导航">',
        '<li><a class="home-link" href="/list/" title="根目录">⌂</a></li>',
    ]
    for crumb in get_breadcrumbs(parent_path):
        name = html_mod.escape(crumb["name"])
        url = html_mod.escape(crumb["url"], quote=True)
        parts.append(f'<li><a href="{url}" title="{name}">{name}</a></li>')
    if current_name:
        name = html_mod.escape(current_name)
        parts.append(f'<li><span class="current" title="{name}">{name}</span></li>')
    parts.append("</ol>")
    return "".join(parts)


@app.get("/list/{file_path:path}", name="list")
async def list_files(request: Request, file_path: Path) -> Response:
    """list files in `file_path`"""
    path = Path(file_path)
    redirect = _canonical_route_redirect(request, "/list", path)
    if redirect:
        return redirect
    if path.is_file():
        return RedirectResponse(url=_file_open_url(path))
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {file_path}")
    template_path = "" if str(file_path) == "." else str(file_path)
    current_name = "根目录" if not template_path else path.name
    breadcrumbs_html = render_breadcrumbs_html(path.parent, current_name)

    return templates.TemplateResponse(
        request,
        "list.html",
        {
            "path": template_path,
            "copy_path": str(path.resolve()),
            "breadcrumbs_html": breadcrumbs_html,
        },
    )


@app.get("/api/list/{file_path:path}", name="api_list")
async def api_list(file_path: str):
    """Return a directory listing and breadcrumbs as JSON."""
    path = Path(file_path)
    if path.is_file():
        raise HTTPException(status_code=409, detail="Requested path is a file, not a directory.")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {file_path}")
    files = get_directory_contents(path)
    return JSONResponse(
        {
            "path": file_path,
            "absolute_path": str(path.resolve()),
            "parent_path": str(path.parent),
            "total": len(files),
            "breadcrumbs": get_breadcrumbs(path),
            "items": files,
        }
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

    url = _route_url("/list", Path(file_path))
    response = RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    return response


@app.get("/excel/{file_path:path}")
async def download_excel(
    file_path: str,
    names: Optional[str] = None,
    header: Optional[bool] = True,
    json_cols: Optional[str] = None,
    delim: str = "\t",
):
    """download file as excel"""
    path = Path(file_path)

    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if path.suffix in {".xls", ".xlsx"}:
        return FileResponse(path)
    names_list = None
    if names:
        names_list = tuple(str(x) for x in names.split(","))
    df = read_file(file_path, names_list, header=header, delim=delim)

    if df is None:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        excel_file_name = path.stem + ".xlsx"
        buf = BytesIO()
        with ExcelWriter(buf, engine="openpyxl") as writer:
            _clean_for_excel(df).to_excel(writer, index=False)
        data = buf.getvalue()
        # 直接用 Response 返回字节，避免 FileResponse 的 Range/206 处理（h11 Content-Length 不一致问题）
        from urllib.parse import quote
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(excel_file_name)}",
                "Content-Length": str(len(data)),
                "Accept-Ranges": "none",
            },
        )
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


def read_file(
    file_path: str,
    names_list: tuple[str, ...] | None = None,
    *,
    header: bool = True,
    delim: str = "\t",
) -> DataFrame | None:
    """以file name为key load&cache文件"""
    if not delim:
        raise HTTPException(status_code=400, detail="delim must not be empty")
    k = (file_path, names_list, header, delim)
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
                    df = pd.read_csv(path, sep=delim, names=list(names_list))
                elif header:
                    df = pd.read_csv(path, sep=delim)
                else:
                    df = pd.read_csv(path, sep=delim, header=None)
                    df.columns = [f"col{i}" for i in range(df.shape[1])]
                # 列名中的 '.' 会被前端 DataTables 当作嵌套路径，统一替换
                df.columns = [str(x).replace(".", "-") for x in df.columns]
            except pd.errors.ParserError:
                # Fallback for ragged files (variable column counts)
                # "Good Taste": Don't crash, handle the data structure as it is (a ragged matrix)
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        lines = [line.rstrip("\n").split(delim) for line in f]
                    
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

                    df.columns = [str(x).replace(".", "-") for x in df.columns]
                    
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


def _build_tsv_context(  # noqa: PLR0913
    request: Request,
    file_path: str,
    start: int | None,
    length: int | None,
    reload: bool | None,
    names: str | None,
    header: bool | None,
    json_link_cols: str | None,
    json_cols: str | None,
    image_cols: str | None,
    hide_cols: str | None,
    delim: str,
) -> dict[str, Any]:
    """Build the shared template context for tabular previews."""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    k = (file_path, tuple(names.split(",")) if names else None, header, delim)
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
        delim=delim,
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

    # Build extra params to pass through to API calls in the template
    extra_params = ""
    if names:
        extra_params += f"&names={names}"
    if not header:
        extra_params += "&header=false"
    from urllib.parse import quote
    extra_params += f"&delim={quote(delim, safe='')}"

    return {
        "request": request,
        "path": file_path,
        "copy_path": str(path.resolve()),
        "parent_path": str(parent_path),
        "name": name,
        "columns": columns,
        "length": length,
        "start": start,
        "recordsTotal": len(df),
        "json_link_cols": json_link_col_list,
        "json_cols": json_col_list,
        "image_cols": image_col_list,
        "hide_cols": hide_col_list,
        "warning_msg": warning_msg,
        "extra_params": extra_params,
        "param_names": names or "",
        "param_header": header if header is not None else True,
        "param_delim": delim,
        "param_json_cols": json_cols or "",
        "param_json_link_cols": json_link_cols or "",
        "param_image_cols": image_cols or "",
        "param_hide_cols": hide_cols or "",
    }


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
    delim: str = "\t",
):
    """Show the primary Tabulator-based tabular page."""
    context = _build_tsv_context(
        request,
        file_path,
        start,
        length,
        reload,
        names,
        header,
        json_link_cols,
        json_cols,
        image_cols,
        hide_cols,
        delim,
    )
    if "error" in context:
        return context
    return templates.TemplateResponse(request, "tsv.html", context)


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


_IMG_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc=)(["\'])(.*?)\2', re.IGNORECASE)


def _rewrite_relative_img_src(html_body: str, base_dir: Path) -> str:
    """Point relative <img src> values at /download/<absolute path>."""
    from urllib.parse import quote, unquote, urlsplit

    def _sub(m: re.Match) -> str:
        src = html_mod.unescape(m.group(3))
        parts = urlsplit(src)
        if parts.scheme or parts.netloc or src.startswith(("/", "#")) or not parts.path:
            return m.group(0)
        target = Path(os.path.normpath(base_dir / unquote(parts.path)))
        return f"{m.group(1)}{m.group(2)}/download/{quote(str(target))}{m.group(2)}"

    return _IMG_SRC_RE.sub(_sub, html_body)


_MD_TABLE_RE = re.compile(r"<table(?:\s[^>]*)?>.*?</table>", re.IGNORECASE | re.DOTALL)


def _wrap_markdown_tables(html_body: str) -> str:
    """Wrap rendered Markdown tables in a keyboard-accessible scroll viewport."""
    table_index = 0

    def _sub(match: re.Match) -> str:
        nonlocal table_index
        table_index += 1
        return (
            f'<div class="markdown-table-scroll" tabindex="0" role="region" '
            f'aria-label="表格 {table_index}，可横向、纵向滚动并调整列宽" '
            f'data-markdown-table="{table_index}">{match.group(0)}</div>'
        )

    return _MD_TABLE_RE.sub(_sub, html_body)


def _mermaid_fence(source, language, css_class, options, md, **kwargs):
    """Emit mermaid source as escaped text; mermaid.js reads it from its HTML-decoded text."""
    return f'<div class="mermaid">{html_mod.escape(source)}</div>'


def _new_md_converter() -> md_lib.Markdown:
    """Markdown converter: math, mermaid and code are separated at parse time."""
    return md_lib.Markdown(
        extensions=[
            "tables",
            "pymdownx.highlight",
            "toc",
            "nl2br",
            "pymdownx.superfences",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "toc": {"title": "目录"},
            # Same markup/CSS hook as the previous codehilite extension.
            "pymdownx.highlight": {"css_class": "codehilite"},
            "pymdownx.superfences": {
                "custom_fences": [
                    {"name": "mermaid", "class": "mermaid", "format": _mermaid_fence},
                ],
            },
            "pymdownx.arithmatex": {"generic": True},
        },
    )


@app.get("/md/{file_path:path}", name="render_md")
async def render_md(request: Request, file_path: str) -> Response:
    """Render a Markdown file as HTML."""
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    async with aiofiles.open(path, encoding="utf-8") as f:
        content = await f.read()

    converter = _new_md_converter()
    html_body = converter.convert(content)

    # Relative image paths would resolve to /md/<dir>/img/x.jpg and hit this
    # route; serve them through /download instead. Exported HTML keeps them as-is.
    if "path" in request.scope and request.query_params.get("download") != "1":
        html_body = _rewrite_relative_img_src(html_body, path.resolve().parent)
    html_body = _wrap_markdown_tables(html_body)

    toc = getattr(converter, "toc", "")  # "" when no headings found

    markdown_breadcrumbs_html = render_breadcrumbs_html(path.parent, path.name)
    markdown_download_url = f"{_route_url('/md', path)}?download=1"
    markdown_raw_download_url = _route_url("/download", path)
    markdown_copy_path = html_mod.escape(str(path.resolve()), quote=True)
    toc_sidebar = (
        '<aside class="toc-sidebar" id="toc-sidebar">'
        '<div class="toc-header">'
        '<div class="toc-title">目录</div>'
        '<button class="toc-sidebar-toggle" type="button" aria-label="折叠目录" title="折叠目录">收起</button>'
        "</div>"
        f"{toc}"
        "</aside>"
        '<div class="toc-resizer" id="toc-resizer" role="separator" aria-label="调整目录宽度" '
        'aria-orientation="vertical" aria-valuemin="180" aria-valuemax="420" aria-valuenow="220" '
        'tabindex="0" title="拖拽调整目录宽度"></div>'
    ) if toc.strip() and "<li>" in toc else ""
    has_toc = bool(toc_sidebar)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{path.name}</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
  <style>
    body {{ background: #f6f8fa; margin: 0; padding: 0; }}
    .topbar {{
      position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 9px 20px; font-size: 13px;
      background: rgba(255,255,255,.97); border-bottom: 1px solid #dde0e4;
      box-shadow: 0 2px 8px rgba(0,0,0,.06); backdrop-filter: blur(8px);
    }}
    .topbar-actions {{ display:flex; flex:0 0 auto; align-items:center; gap:8px; }}
    .topbar-actions a, .topbar-actions button {{
      padding: 4px 8px; color: #1a56a0; background: #fff; border: 1px solid #c9d1d9;
      border-radius: 4px; text-decoration: none; white-space: nowrap;
      font: inherit; cursor: pointer;
    }}
    .topbar-actions a:hover, .topbar-actions button:hover {{ border-color:#1a56a0; text-decoration:none; }}
    .breadcrumbs {{
      display:flex; flex:1 1 auto; align-items:center; min-width:0;
      margin:0; padding:0; overflow-x:auto; overflow-y:hidden;
      list-style:none; white-space:nowrap; scrollbar-width:thin;
    }}
    .breadcrumbs li {{ display:flex; flex:0 0 auto; align-items:center; min-width:0; }}
    .breadcrumbs li + li::before {{ content:"/"; margin:0 7px; color:#a0a7b1; }}
    .breadcrumbs a, .breadcrumbs .current {{
      overflow:hidden; max-width:220px; text-overflow:ellipsis; white-space:nowrap;
    }}
    .breadcrumbs a {{ color:#1a56a0; text-decoration:none; }}
    .breadcrumbs a:hover {{ text-decoration:underline; }}
    .breadcrumbs .current {{ color:#344054; font-weight:600; }}
    .home-link {{
      display:inline-flex; align-items:center; justify-content:center;
      width:28px; height:28px; border-radius:5px; background:#f1f5f9; text-decoration:none;
    }}
    .toc-toggle {{
      border: 1px solid #c9d1d9; border-radius: 4px; background: #fff;
      color: #1a56a0; cursor: pointer; font-size: 12px; padding: 3px 8px;
      margin-left: 0;
    }}
    .toc-toggle:hover {{ background: #f3f6f8; }}
    .page-layout {{
      display: flex;
      max-width: {'1200px' if has_toc else '960px'};
      margin: 24px auto;
      gap: 24px;
      padding: 0 20px;
      align-items: flex-start;
    }}
    .markdown-body {{
      background: #fff;
      flex: 1;
      min-width: 0;
      padding: 32px 40px;
      border: 1px solid #dde0e4;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    /* Markdown tables get a dedicated two-axis scroll viewport and sticky header. */
    .markdown-table-scroll {{
      position: relative;
      display: block;
      width: 100%;
      max-width: 100%;
      max-height: 80vh;
      margin: 16px 0;
      overflow: auto;
      overscroll-behavior: contain;
      background: #fff;
      border: 1px solid #d0d7de;
      border-radius: 6px;
    }}
    .markdown-table-scroll:focus {{
      outline: 2px solid #0969da;
      outline-offset: 2px;
    }}
    .markdown-table-scroll > table {{
      display: table;
      width: max-content;
      min-width: 100%;
      max-width: none;
      max-height: none;
      margin: 0;
      overflow: visible;
      border-collapse: collapse;
      border-spacing: 0;
    }}
    .markdown-table-scroll > table.grip-flex:not([style]) {{
      width: max-content !important;
      min-width: 100% !important;
    }}
    .markdown-table-scroll th,
    .markdown-table-scroll td {{
      min-width: 6.5rem;
      overflow-wrap: break-word;
      white-space: normal;
    }}
    .markdown-table-scroll thead th {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: #f6f8fa;
      box-shadow: inset 0 -1px 0 #d0d7de;
    }}
    .markdown-table-scroll .grip-container {{
      z-index: 4;
    }}
    .markdown-table-scroll .grip-handle:hover .markdown-column-grip,
    .markdown-table-scroll .grip-drag .markdown-column-grip {{
      background: #0969da;
    }}
    .markdown-column-grip {{
      width: 2px;
      height: 100%;
      margin: 0 auto;
      background: transparent;
    }}
    /* TOC sidebar */
    .toc-sidebar {{
      width: var(--toc-width, 220px);
      flex-shrink: 0;
      background: #fff;
      border: 1px solid #dde0e4;
      border-radius: 6px;
      padding: 14px 16px;
      position: sticky;
      top: 48px;
      max-height: calc(100vh - 72px);
      overflow-y: auto;
      font-size: 13px;
    }}
    .toc-header {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px; margin-bottom: 8px;
    }}
    .toc-title {{ font-weight: 600; color: #333; }}
    .toc-sidebar-toggle {{
      border: 0; border-radius: 4px; background: transparent; color: #57606a;
      cursor: pointer; font-size: 11px; padding: 3px 5px;
    }}
    .toc-sidebar-toggle:hover {{ background: #f3f6f8; color: #1a56a0; }}
    .toc-resizer {{
      align-self: stretch; flex: 0 0 8px; cursor: col-resize; position: relative;
      margin: 0 -12px 0 -12px; z-index: 2; touch-action: none;
    }}
    .toc-resizer::after {{
      content: ""; position: absolute; top: 0; bottom: 0; left: 3px; width: 2px;
      background: transparent; transition: background .15s;
    }}
    .toc-resizer:hover::after, .toc-resizer:focus::after, body.toc-resizing .toc-resizer::after {{
      background: #8c959f;
    }}
    body.toc-resizing {{ cursor: col-resize; user-select: none; }}
    .toc-collapsed .toc-sidebar, .toc-collapsed .toc-resizer {{ display: none; }}
    .toc-sidebar .toc ul {{ list-style: none; padding-left: 12px; margin: 0; }}
    .toc-sidebar .toc > ul {{ padding-left: 0; }}
    .toc-sidebar .toc li {{ margin: 4px 0; }}
    .toc-sidebar .toc a {{ color: #1a56a0; text-decoration: none; }}
    .toc-sidebar .toc a:hover {{ text-decoration: underline; }}
    /* Code block copy button */
    .code-wrapper {{ position: relative; }}
    .copy-btn {{
      position: absolute; top: 6px; right: 8px;
      background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;
      padding: 2px 8px; font-size: 11px; cursor: pointer; opacity: 0;
      transition: opacity 0.15s;
    }}
    .code-wrapper:hover .copy-btn {{ opacity: 1; }}
    .copy-btn.copied {{ background: #d4edda; border-color: #97c8a0; color: #2d6a4f; }}
    .markdown-body .mermaid svg {{ cursor: zoom-in; }}
    .mm-modal {{
      position: fixed; inset: 0; z-index: 1000; display: none;
      background: rgba(0,0,0,0.55);
    }}
    .mm-modal.open {{ display: block; }}
    .mm-stage {{
      position: absolute; inset: 40px; background: #fff; border-radius: 8px;
      overflow: hidden; cursor: grab; touch-action: none;
    }}
    .mm-stage.dragging {{ cursor: grabbing; }}
    .mm-canvas {{ position: absolute; left: 0; top: 0; transform-origin: 0 0; }}
    .mm-canvas svg {{ max-width: none !important; display: block; }}
    .mm-tools {{
      position: absolute; top: 48px; right: 52px; z-index: 1; display: flex; gap: 6px;
    }}
    .mm-tools button {{
      min-width: 34px; height: 30px; border: 1px solid #ccc; border-radius: 4px;
      background: #fff; cursor: pointer; font-size: 14px;
    }}
    .mm-tools .mm-scale {{ align-self: center; min-width: 48px; text-align: center; font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <div class="topbar">
    {markdown_breadcrumbs_html}
    <div class="topbar-actions">
      {'<button class="toc-toggle" type="button" aria-expanded="true">隐藏目录</button>' if has_toc else ''}
      <button class="copy-path-button" type="button" data-copy-path="{markdown_copy_path}" title="{markdown_copy_path}">复制路径</button>
      <a href="{markdown_download_url}">下载 HTML</a>
      <a href="{markdown_raw_download_url}">下载</a>
    </div>
  </div>
  <script>
    (function() {{
      var breadcrumbs = document.querySelector('.breadcrumbs');
      window.requestAnimationFrame(function() {{
        breadcrumbs.scrollLeft = breadcrumbs.scrollWidth;
      }});
      var copyButton = document.querySelector('.copy-path-button');
      copyButton.addEventListener('click', function() {{
        var value = copyButton.getAttribute('data-copy-path');
        var done = function() {{
          copyButton.textContent = '已复制';
          window.setTimeout(function() {{ copyButton.textContent = '复制路径'; }}, 1400);
        }};
        var fallback = function() {{
          var textarea = document.createElement('textarea');
          textarea.value = value;
          textarea.style.cssText = 'position:fixed;left:-9999px;top:0';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          textarea.remove();
          done();
        }};
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(value).then(done, fallback);
        }} else {{
          fallback();
        }}
      }});
    }})();
  </script>
  <div class="page-layout" id="page-layout">
    {toc_sidebar}
    <article class="markdown-body">
{html_body}
    </article>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/column-resizer@1.4.0/dist/column-resizer.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.0/mermaid.min.js"></script>
  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
  </script>
  <script>
    (function() {{
      var ColumnResizerClass = window.ColumnResizer && window.ColumnResizer.default;
      if (!ColumnResizerClass) return;
      var measureContext = document.createElement('canvas').getContext('2d');
      function measureText(value, bold) {{
        measureContext.font = (bold ? '600 ' : '') + '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        return measureContext.measureText(value || '').width;
      }}
      function initialWidths(table) {{
        var headers = table.tHead ? Array.from(table.tHead.rows[0].cells) : [];
        var rows = table.tBodies.length ? Array.from(table.tBodies[0].rows).slice(0, 100) : [];
        return headers.map(function(header, index) {{
          var values = [header.innerText.replace(/\\s+/g, ' ').trim()];
          rows.forEach(function(row) {{
            var cell = row.cells[index];
            if (cell) values.push(cell.innerText.replace(/\\s+/g, ' ').trim());
          }});
          var naturalWidth = Math.max.apply(null, values.map(function(value) {{
            return measureText(value, value === values[0]);
          }}));
          var compact = values.slice(1).filter(Boolean).every(function(value) {{
            return /^[\\d\\s.,:+\\-%/]+$/.test(value) || /^\\d{{4}}-\\d{{2}}-\\d{{2}}/.test(value);
          }});
          var cap = compact ? 170 : 420;
          return Math.round(Math.max(96, Math.min(cap, naturalWidth + 30)));
        }});
      }}
      document.querySelectorAll('.markdown-table-scroll > table').forEach(function(table, index) {{
        table.id = 'markdown-table-' + (index + 1);
        table.classList.add('markdown-table');
        var options = {{
          resizeMode: 'overflow',
          liveDrag: true,
          minWidth: 80,
          headerOnly: true,
          removePadding: false,
          gripInnerHtml: '<div class="markdown-column-grip"></div>',
          serialize: true
        }};
        try {{
          if (!sessionStorage.getItem(table.id)) options.widths = initialWidths(table);
        }} catch (_error) {{
          options.widths = initialWidths(table);
        }}
        new ColumnResizerClass(table, options);
      }});
    }})();
  </script>
  <div class="mm-modal" id="mm-modal" aria-hidden="true">
    <div class="mm-tools">
      <button type="button" data-mm="out" title="缩小">−</button>
      <span class="mm-scale" id="mm-scale">100%</span>
      <button type="button" data-mm="in" title="放大">+</button>
      <button type="button" data-mm="fit" title="适应窗口">适应</button>
      <button type="button" data-mm="close" title="关闭 (Esc)">✕</button>
    </div>
    <div class="mm-stage" id="mm-stage"><div class="mm-canvas" id="mm-canvas"></div></div>
  </div>
  <script>
    // Click a rendered mermaid diagram to open it in a zoom/pan viewer.
    (function() {{
      var modal = document.getElementById('mm-modal');
      var stage = document.getElementById('mm-stage');
      var canvas = document.getElementById('mm-canvas');
      var scaleLabel = document.getElementById('mm-scale');
      var s = 1, x = 0, y = 0, baseW = 0, baseH = 0;

      function apply() {{
        canvas.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + s + ')';
        scaleLabel.textContent = Math.round(s * 100) + '%';
      }}
      function zoomAt(factor, cx, cy) {{
        var ns = Math.min(20, Math.max(0.1, s * factor));
        x = cx - (cx - x) * ns / s;
        y = cy - (cy - y) * ns / s;
        s = ns;
        apply();
      }}
      function fit() {{
        var w = stage.clientWidth, h = stage.clientHeight;
        s = Math.min(w / baseW, h / baseH) * 0.95;
        x = (w - baseW * s) / 2;
        y = (h - baseH * s) / 2;
        apply();
      }}
      function open(svg) {{
        // Use the diagram's natural size (viewBox) so 100% means 1:1.
        var vb = svg.viewBox && svg.viewBox.baseVal;
        var box = (vb && vb.width) ? vb : svg.getBoundingClientRect();
        var clone = svg.cloneNode(true);
        clone.removeAttribute('style');
        baseW = box.width || 800;
        baseH = box.height || 600;
        clone.setAttribute('width', baseW);
        clone.setAttribute('height', baseH);
        canvas.innerHTML = '';
        canvas.appendChild(clone);
        modal.classList.add('open');
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        fit();
      }}
      function close() {{
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        canvas.innerHTML = '';
      }}

      document.addEventListener('click', function(e) {{
        var svg = e.target.closest && e.target.closest('.markdown-body .mermaid svg');
        if (svg) open(svg);
      }});
      modal.addEventListener('click', function(e) {{
        var btn = e.target.closest('button[data-mm]');
        if (btn) {{
          var act = btn.getAttribute('data-mm');
          var cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
          if (act === 'in') zoomAt(1.25, cx, cy);
          else if (act === 'out') zoomAt(0.8, cx, cy);
          else if (act === 'fit') fit();
          else close();
        }} else if (e.target === modal) {{
          close();
        }}
      }});
      document.addEventListener('keydown', function(e) {{
        if (!modal.classList.contains('open')) return;
        if (e.key === 'Escape') close();
        else if (e.key === '+' || e.key === '=') zoomAt(1.25, stage.clientWidth / 2, stage.clientHeight / 2);
        else if (e.key === '-') zoomAt(0.8, stage.clientWidth / 2, stage.clientHeight / 2);
        else if (e.key === '0') fit();
      }});
      stage.addEventListener('wheel', function(e) {{
        e.preventDefault();
        var r = stage.getBoundingClientRect();
        zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
      }}, {{ passive: false }});

      var dragId = null, lastX = 0, lastY = 0;
      stage.addEventListener('pointerdown', function(e) {{
        dragId = e.pointerId; lastX = e.clientX; lastY = e.clientY;
        stage.setPointerCapture(dragId);
        stage.classList.add('dragging');
      }});
      stage.addEventListener('pointermove', function(e) {{
        if (e.pointerId !== dragId) return;
        x += e.clientX - lastX; y += e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        apply();
      }});
      function endDrag(e) {{
        if (e.pointerId !== dragId) return;
        dragId = null;
        stage.classList.remove('dragging');
      }}
      stage.addEventListener('pointerup', endDrag);
      stage.addEventListener('pointercancel', endDrag);
      stage.addEventListener('dblclick', fit);
    }})();
  </script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <script>
    // Math was identified by the Markdown parser (pymdownx.arithmatex); only
    // render those elements so mermaid and code blocks are never touched.
    document.querySelectorAll('.arithmatex').forEach(function(el) {{
      var tex = el.textContent.trim();
      var display = el.tagName === 'DIV';
      tex = tex.replace(/^\\\\[\\[(]/, '').replace(/\\\\[\\])]$/, '');
      try {{
        katex.render(tex, el, {{ displayMode: display, throwOnError: false }});
      }} catch (_) {{}}
    }});
  </script>
  <script>
    // Keep the directory panel comfortable across sessions and viewport sizes.
    (function() {{
      var layout = document.getElementById('page-layout');
      var sidebar = document.getElementById('toc-sidebar');
      var resizer = document.getElementById('toc-resizer');
      var toggle = document.querySelector('.toc-toggle');
      var sidebarToggle = document.querySelector('.toc-sidebar-toggle');
      if (!layout || !sidebar || !resizer || !toggle || !sidebarToggle) return;

      var minWidth = 180;
      var maxWidth = 420;
      var storageKey = 'fserver-md-toc';
      var saved = null;
      try {{ saved = JSON.parse(localStorage.getItem(storageKey) || 'null'); }} catch (_) {{}}

      function clamp(width) {{
        return Math.max(minWidth, Math.min(maxWidth, width));
      }}
      function setWidth(width) {{
        width = clamp(width);
        layout.style.setProperty('--toc-width', width + 'px');
        resizer.setAttribute('aria-valuenow', String(width));
        if (saved) saved.width = width;
        try {{ localStorage.setItem(storageKey, JSON.stringify({{ width: width, collapsed: layout.classList.contains('toc-collapsed') }})); }} catch (_) {{}}
      }}
      function setCollapsed(collapsed) {{
        layout.classList.toggle('toc-collapsed', collapsed);
        toggle.textContent = collapsed ? '显示目录' : '隐藏目录';
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.setAttribute('aria-label', collapsed ? '显示目录' : '隐藏目录');
        sidebarToggle.textContent = collapsed ? '展开' : '收起';
        sidebarToggle.setAttribute('aria-label', collapsed ? '展开目录' : '折叠目录');
        try {{ localStorage.setItem(storageKey, JSON.stringify({{ width: parseInt(getComputedStyle(layout).getPropertyValue('--toc-width')) || 220, collapsed: collapsed }})); }} catch (_) {{}}
      }}

      setWidth(saved && Number.isFinite(saved.width) ? saved.width : 220);
      setCollapsed(Boolean(saved && saved.collapsed));
      toggle.addEventListener('click', function() {{ setCollapsed(!layout.classList.contains('toc-collapsed')); }});
      sidebarToggle.addEventListener('click', function() {{ setCollapsed(!layout.classList.contains('toc-collapsed')); }});

      var startX = 0;
      var startWidth = 0;
      function stopResize() {{
        if (!document.body.classList.contains('toc-resizing')) return;
        document.body.classList.remove('toc-resizing');
        document.removeEventListener('pointermove', resize);
        document.removeEventListener('pointerup', stopResize);
        document.removeEventListener('pointercancel', stopResize);
      }}
      function resize(event) {{
        setWidth(startWidth + event.clientX - startX);
      }}
      resizer.addEventListener('pointerdown', function(event) {{
        if (layout.classList.contains('toc-collapsed')) return;
        startX = event.clientX;
        startWidth = sidebar.getBoundingClientRect().width;
        document.body.classList.add('toc-resizing');
        document.addEventListener('pointermove', resize);
        document.addEventListener('pointerup', stopResize);
        document.addEventListener('pointercancel', stopResize);
        event.preventDefault();
      }});
      resizer.addEventListener('keydown', function(event) {{
        var width = sidebar.getBoundingClientRect().width;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {{
          setWidth(width + (event.key === 'ArrowRight' ? 10 : -10));
          event.preventDefault();
        }} else if (event.key === 'Enter' || event.key === ' ') {{
          setCollapsed(!layout.classList.contains('toc-collapsed'));
          event.preventDefault();
        }}
      }});
      window.addEventListener('resize', function() {{
        if (window.innerWidth <= 760) setCollapsed(false);
      }});
    }})();
  </script>
  <style>
    @media (max-width: 760px) {{
      .topbar {{ align-items: flex-start; flex-wrap: wrap; padding: 8px; }}
      .breadcrumbs {{ width: 100%; }}
      .breadcrumbs a, .breadcrumbs .current {{ max-width: 130px; }}
      .topbar-actions {{ width: 100%; justify-content: flex-end; }}
      .page-layout {{ display: block; margin: 12px auto; padding: 0 10px; }}
      .toc-sidebar {{ width: auto; position: static; max-height: none; margin-bottom: 12px; }}
      .toc-resizer {{ display: none; }}
      .markdown-body {{ padding: 22px 18px; }}
      .markdown-table-scroll {{ max-height: 65vh; }}
      .toc-toggle {{ margin-left: 6px; }}
    }}
    /* Images: keep thumbnails readable in narrow table cells; click to enlarge */
    .markdown-body img {{ cursor: zoom-in; }}
    .img-lightbox {{
      position: fixed; inset: 0; z-index: 1000; display: none;
      align-items: center; justify-content: center;
      background: rgba(0, 0, 0, 0.82); cursor: zoom-out;
    }}
    .img-lightbox.open {{ display: flex; }}
    .img-lightbox img {{
      max-width: 94vw; max-height: 94vh; background: #fff;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5); cursor: zoom-out;
    }}
  </style>
  <script>
    // Inject copy buttons into every <pre><code> block
    document.querySelectorAll('pre').forEach(function(pre) {{
      var wrapper = document.createElement('div');
      wrapper.className = 'code-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = '复制';
      wrapper.appendChild(btn);
      btn.addEventListener('click', function() {{
        var code = pre.querySelector('code') || pre;
        var text = code.innerText;
        function onSuccess() {{
          btn.textContent = '已复制';
          btn.classList.add('copied');
          setTimeout(function() {{ btn.textContent = '复制'; btn.classList.remove('copied'); }}, 1500);
        }}
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(text).then(onSuccess);
        }} else {{
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
          document.body.appendChild(ta);
          ta.focus(); ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          onSuccess();
        }}
      }});
    }});
  </script>
  <script>
    // Images: enforce a minimum display size (never above natural size) and
    // open a lightbox on click. A following "原图" link is used as the large
    // version when it loads; otherwise the local image is shown.
    (function() {{
      var MIN_W = 140;
      var imgs = document.querySelectorAll('.markdown-body img');
      function applyMin(img) {{
        if (img.naturalWidth > 48) img.style.minWidth = Math.min(img.naturalWidth, MIN_W) + 'px';
      }}
      var box = document.createElement('div');
      box.className = 'img-lightbox';
      var big = document.createElement('img');
      box.appendChild(big);
      document.body.appendChild(box);
      function close() {{ box.classList.remove('open'); big.removeAttribute('src'); }}
      box.addEventListener('click', close);
      document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') close(); }});
      imgs.forEach(function(img) {{
        if (img.complete) applyMin(img); else img.addEventListener('load', function() {{ applyMin(img); }});
        img.addEventListener('click', function(e) {{
          e.preventDefault();
          big.src = img.currentSrc || img.src;
          box.classList.add('open');
          var a = img.nextElementSibling;
          if (a && a.tagName === 'A' && a.textContent.indexOf('原图') !== -1) {{
            var hi = new Image();
            hi.onload = function() {{ if (box.classList.contains('open')) big.src = hi.src; }};
            hi.src = a.href;
          }}
        }});
      }});
    }})();
  </script>
</body>
</html>"""
    response = Response(content=html, media_type="text/html")
    if request.query_params.get("download") == "1":
        from urllib.parse import quote

        response.headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(path.stem + '.html')}"
    return response


async def render_markdown_file(path: Path) -> str:
    """Render a local Markdown file using the same document as the /md route."""
    request = Request({"type": "http", "query_string": b""})
    response = await render_md(request, str(path))
    return bytes(response.body).decode("utf-8")


@app.get("/api/tsv/key/{file_path:path}")
async def api_tsv_key(
    request: Request,
    file_path: str,
    key: str,
    reload: Optional[bool] = False,
    names: Optional[str] = None,
    header: Optional[bool] = True,
    delim: str = "\t",
):
    """tsv html get keys of tsv file"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    names_list = tuple(names.split(",")) if names else None
    if reload and (file_path, names_list, header, delim) in cache:
        del cache[(file_path, names_list, header, delim)]

    df = read_file(file_path, names_list, header=header, delim=delim)
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
    names: Optional[str] = None,
    header: Optional[bool] = True,
    json_link_cols: Optional[str] = None,
    json_cols: Optional[str] = None,
    delim: str = "\t",
):
    """return table data as json for datatables ajax call"""
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}

    names_list = tuple(names.split(",")) if names else None
    if reload and (file_path, names_list, header, delim) in cache:
        del cache[(file_path, names_list, header, delim)]

    df = read_file(file_path, names_list, header=header, delim=delim)

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
    delim: str = "\t",
) -> JSONResponse:
    """Retrieve raw content of a specific cell in a TSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    names_list = tuple(names.split(",")) if names else None

    # Read the file with no json_cols processed to get raw content
    df = read_file(file_path, names_list, header=header, delim=delim)  # Ensure raw content

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
