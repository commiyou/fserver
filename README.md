# fserver

[English](README.en.md) | 中文

面向工程师的轻量级文件服务器与数据查看器，提供目录浏览、文件上传、表格查看、数据导出和 Markdown 渲染。

## 功能

### 文件管理

- 浏览本地目录和文件，目录路径与 URL 直接对应。
- 将文件拖拽或通过 API 上传到指定目录。
- 下载原始文件。
- 将 TSV、CSV、JSON Lines 等表格文件转换为 Excel。

### 表格查看器

- 支持 TSV、CSV、JSON Lines 和 XLSX。
- 服务端分页，适合查看大型文件。
- 按列和值搜索和筛选。
- 隐藏或显示列。
- 自动识别并格式化单元格中的 JSON。
- 自动识别图片 URL 并渲染预览。
- 对列数不规则的 TSV/CSV 启用兼容模式，并显示提示。

### Markdown 查看器

- 支持 GitHub 风格 Markdown、表格、代码高亮和 Mermaid 图表。
- 自动生成目录侧边栏。
- 目录侧边栏支持拖拽调整宽度、折叠和展开。
- 侧边栏宽度和折叠状态会保存在浏览器 `localStorage` 中。
- 代码块提供复制按钮。
- 支持下载包含上述交互能力的 HTML 文件。

## 快速开始

项目使用 `uv` 管理 Python 依赖，需要 Python 3.12 或更高版本。

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone <repo_url>
cd fserver

# 创建并激活虚拟环境
uv venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 安装依赖
uv pip install -r requirements.txt
```

启动开发服务器：

```bash
uvicorn fserver:app --host 0.0.0.0 --port 8000 --reload
```

打开 `http://localhost:8000/list/`。

也可以直接运行 Python 文件；这种方式默认使用 `8113` 端口：

```bash
python fserver.py
```

## `fp` 地址命令

安装项目后可以使用 `fp` 为文件或目录生成 fserver 地址。命令始终使用输入路径的绝对路径：

```bash
uv pip install -e .
source .venv/bin/activate
fp README.md
fp data.tsv some-directory
```

Markdown 文件会输出预览和交互式 HTML 下载地址；表格文件会输出表格查看和 Excel 地址；目录只输出目录浏览地址。

如果不想激活虚拟环境，也可以直接调用命令：

```bash
/ssd1/youbin/src/fserver/.venv/bin/fp README.md
```

或者使用 `uv run`：

```bash
uv run fp README.md
uv run --project /ssd1/youbin/src/fserver fp /path/to/README.md
```

如果希望在任意目录直接输入 `fp`，可以将虚拟环境的 `bin` 目录加入 `PATH`：

```bash
echo 'export PATH="/ssd1/youbin/src/fserver/.venv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
fp README.md
```

```bash
fp --host bddwd-acg-tge43qlalf9.bddwd.baidu.com --port 8113 README.md
fp --format json README.md
```

可通过 `--host`、`--port` 和 `--scheme` 覆盖服务器地址，默认值为本机 FQDN、`8113` 和 `http`。

## API

### 浏览和查看

- `GET /list/{path}`：浏览目录；如果路径是文件，则跳转到对应查看器。
- `GET /tsv/{path}`：表格查看器。
  - `start`：起始行，默认 `0`。
  - `length`：返回行数，默认 `1000`。
  - `key` / `value`：按列名和值筛选。
  - `names`：以逗号分隔的自定义列名。
  - `header`：是否将首行作为表头。
  - `json_cols`：需要格式化 JSON 的列。
  - `json_link_cols`：需要渲染 JSON 链接的列。
  - `image_cols`：需要渲染图片预览的列。
  - `hide_cols`：默认隐藏的列。
- `GET /md/{path}`：渲染 Markdown 文件。
- `GET /md/{path}?download=1`：下载包含交互能力的 HTML。
- `GET /txt/{path}`：以纯文本查看文件。

### 下载和导出

- `GET /download/{path}`：下载原始文件。
- `GET /excel/{path}`：将表格文件转换为 Excel 后下载。

### 工具接口

- `GET /api/tsv/key/{path}`：获取指定列的唯一值。
- `GET /json_viewer`：打开独立的 JSON 格式化查看页面。

## 架构

- 后端：Python、FastAPI、Pandas、Aiofiles。
- 模板：Jinja2 和原生 HTML。
- 表格交互：jQuery DataTables 及其扩展。
- Markdown：Python-Markdown、代码高亮、Mermaid。
- 缓存：使用 `cachetools` 缓存已读取的 DataFrame，默认缓存 48 小时。

## 注意事项

- 文件路径相对于服务进程的当前工作目录。
- 下载的 Markdown HTML 使用 CDN 加载 Mermaid、代码高亮和 GitHub Markdown 样式；离线环境下这些外部资源可能无法加载。
- 当前大文件读取仍会将 DataFrame 加载到内存，超大文件建议关注进程内存使用。

## 待办事项

- [ ] 流式读取超大文件，降低内存占用。
- [ ] 支持简单的文本文件在线编辑。
- [ ] 提供 Docker 镜像。
