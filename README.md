# fserver

一个简洁的文件服务器，提供文件管理和数据查看的增强功能。

## 功能特性

-   **文件浏览**: 轻松浏览目录和文件内容。
-   **文件上传**: 方便地将文件上传到指定目录。
-   **Excel 导出**: 支持将多种表格数据（TSV, CSV, JSON, NDJSON）转换为 Excel 格式并下载。
-   **TSV/CSV 在线查看**: 提供交互式网页版表格数据查看器，支持分页、数据过滤以及 JSON 列的特殊显示处理。
-   **SQLite 数据库查询**: 直接查询 SQLite 数据库并显示结果。
-   **JSON 内容美化**: 提供 JSON 内容的格式化和高亮显示。

## 页面交互端点概览

以下是 `fserver` 提供的主要具有页面交互的端点及其功能和参数说明：

### 1. `GET /list/{file_path:path}`

*   **功能**: 列出指定路径下的文件和目录。若 `file_path` 为文件，则重定向至该文件的 TSV 查看器。
*   **参数**:
    *   `file_path` (路径参数, `Path`): 目录或文件的 URL 编码路径。

### 2. `POST /upload/{file_path:path}`

*   **功能**: 将文件上传到指定目录。
*   **参数**:
    *   `file_path` (路径参数, `str`): 目标目录的 URL 编码路径。
    *   `file` (请求体文件, `UploadFile`): 待上传的文件。

### 3. `GET /tsv/{file_path:path}`

*   **功能**: 渲染交互式 HTML 页面以查看表格数据。支持分页、过滤及 JSON 列显示。
*   **参数**:
    *   `file_path` (路径参数, `str`): 表格数据文件的 URL 编码路径。
    *   `start` (查询参数, 可选, `int`, 默认值: `0`): 数据显示起始行索引（用于分页）。
    *   `length` (查询参数, 可选, `int`, 默认值: `1000`): 每页显示行数。
    *   `reload` (查询参数, 可选, `bool`, 默认值: `False`): 若为 `True`，则强制从磁盘重新加载文件，忽略缓存。
    *   `key` (查询参数, 可选, `str`): 用于过滤数据的列名。
    *   `value` (查询参数, 可选, `str`): 在 `key` 列中匹配的值。
    *   `names` (查询参数, 可选, `str`): 逗号分隔的列名列表，适用于无标题行的文件。
    *   `header` (查询参数, 可选, `bool`, 默认值: `True`): 指示文件是否包含标题行。
    *   `json_link_cols` (查询参数, 可选, `str`): 逗号分隔的 0 索引列号列表，其 JSON 内容将显示为指向 `/json_viewer` 的可点击链接。
    *   `json_cols` (查询参数, 可选, `str`): 逗号分隔的 0 索引列号列表，其 JSON 内容将直接在表格单元格中美化打印。

### 4. `GET /json_viewer`

*   **功能**: 在网页中以格式化和可读方式显示提供的 JSON 内容。
*   **参数**:
    *   `content` (查询参数, `str`): 待显示的 URL 编码 JSON 字符串。

## 安装与部署 (Installation & Deployment)

本项目使用 `uv` 进行依赖管理和虚拟环境创建，并推荐使用 `uvicorn` 作为 ASGI 服务器，生产环境可结合 `gunicorn` 和反向代理。

### 1. 安装 uv

如果你的系统上尚未安装 `uv`，可以通过以下方式安装：

```bash
# 对于 macOS 和 Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 对于 Windows (使用 PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

安装完成后，请确保 `uv` 可执行文件已添加到你的系统 PATH 中（可能需要重启终端）。

### 2. 创建并激活虚拟环境

进入项目根目录，使用 `uv` 创建虚拟环境并安装依赖：

```bash
cd /path/to/fserver  # 将此路径替换为你的项目实际路径
uv venv
source .venv/bin/activate  # macOS/Linux 用户
# 或 .venv\\Scripts\\activate # Windows 用户
```

### 3. 安装项目依赖

激活虚拟环境后，安装 `requirements.txt` 中定义的所有依赖项：

```bash
uv pip install -r requirements.txt
```

### 4. 运行应用

可以直接使用 `uvicorn` 运行应用：

```bash
uvicorn fserver:app --host 0.0.0.0 --port 8000 --reload
```

应用将在 `http://0.0.0.0:8000` 上运行。`--reload` 标志会在代码更改时自动重启服务器，方便开发调试。
API 文档地址: `http://0.0.0.0:8000/docs` 和 `http://0.0.0.0:8000/redoc`。
