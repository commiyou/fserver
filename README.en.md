# fserver

[中文](README.md) | English

A lightweight file server and data viewer for engineers. It provides directory browsing, file uploads, tabular data inspection, exports, and Markdown rendering.

## Features

### File management

- Browse local directories and files with URL paths that map directly to the filesystem.
- Upload files to a target directory through the UI or API.
- Download original files.
- Convert TSV, CSV, and JSON Lines data into Excel workbooks.

### Tabular viewer

- Supports TSV, CSV, JSON Lines, and XLSX.
- Server-side pagination for large files.
- Search and filter by column and value.
- Hide and show columns.
- Automatically format JSON values inside cells.
- Render image URL previews.
- Load ragged TSV/CSV files in compatibility mode and show a warning.

### Markdown viewer

- Render GitHub-style Markdown with tables, code highlighting, and Mermaid diagrams.
- Generate a table-of-contents sidebar automatically.
- Resize the sidebar by dragging, or collapse and expand it.
- Persist sidebar width and collapsed state in browser `localStorage`.
- Add copy buttons to code blocks.
- Download a rendered HTML document that keeps these interactions.

## Quick start

The project uses `uv` for dependency management and requires Python 3.12 or newer.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project
git clone <repo_url>
cd fserver

# Create and activate a virtual environment
uv venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
uv pip install -r requirements.txt
```

Start the development server:

```bash
uvicorn fserver:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/list/`.

You can also run the Python file directly. That mode defaults to port `8113`:

```bash
python fserver.py
```

## API

### Browse and view

- `GET /list/{path}`: Browse a directory. File paths redirect to the appropriate viewer.
- `GET /tsv/{path}`: Open the tabular viewer.
  - `start`: Starting row, default `0`.
  - `length`: Number of rows, default `1000`.
  - `key` / `value`: Filter by column name and value.
  - `names`: Comma-separated custom column names.
  - `header`: Whether the first row is a header.
  - `json_cols`: Columns to format as JSON.
  - `json_link_cols`: Columns to render as JSON links.
  - `image_cols`: Columns to render as image previews.
  - `hide_cols`: Columns hidden by default.
- `GET /md/{path}`: Render a Markdown file.
- `GET /md/{path}?download=1`: Download the rendered interactive HTML.
- `GET /txt/{path}`: View a file as plain text.

### Download and export

- `GET /download/{path}`: Download the original file.
- `GET /excel/{path}`: Convert tabular data to an Excel download.

### Utility endpoints

- `GET /api/tsv/key/{path}`: Return unique values for a column.
- `GET /json_viewer`: Open the standalone JSON formatter.

## Architecture

- Backend: Python, FastAPI, Pandas, and Aiofiles.
- Templates: Jinja2 and native HTML.
- Table interactions: jQuery DataTables and extensions.
- Markdown: Python-Markdown, code highlighting, and Mermaid.
- Caching: `cachetools` caches loaded DataFrames for 48 hours by default.

## Notes

- File paths are relative to the service process's current working directory.
- Downloaded Markdown HTML loads Mermaid, syntax highlighting, and GitHub Markdown styles from CDNs. Those resources may not load in an offline environment.
- Large files are currently loaded into memory as DataFrames. Monitor process memory for very large inputs.

## Roadmap

- [ ] Stream very large files to reduce memory usage.
- [ ] Add simple in-browser text editing.
- [ ] Provide a Docker image.
