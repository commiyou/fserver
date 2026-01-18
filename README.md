# fserver
==以下gemini cli撰写==

专为工程师设计的高性能文件服务器与数据查看器。
以**极速**和**好品味 (Good Taste)** 为核心，拒绝繁杂，提供直观、纯粹的数据浏览体验。

## 核心理念 (Philosophy)

*   **Fail Fast**: 尽可能直接访问数据，拒绝过度封装。
*   **Data Structures > Control Flow**: 通过优良的数据结构设计消除复杂的控制逻辑。
*   **Minimalism**: 后端 FastAPI，前端原生 jQuery + DataTables，无重型框架负担。

## 核心功能 (Features)

### 📂 文件管理
*   **极速浏览**: 像本地文件系统一样浏览远程目录。
*   **便捷上传**: 支持直接拖拽或 API 上传文件到任意目录。
*   **一键导出**: 将任意表格数据（TSV, CSV, JSON Lines）一键转换为 **Excel** 格式下载。

### 📊 数据可视化 (Tabular Viewer)
针对 TSV/CSV/JSONL 文件的专业级查看器：
*   **智能解析**:
    *   自动处理 TSV/CSV 格式。
    *   **鲁棒性**: 遇到列数不一致的“脏数据”自动启用兼容模式加载，并给出醒目提示。
*   **强大交互**:
    *   **服务端分页**: 轻松处理 GB 级大文件，毫秒级响应。
    *   **全文检索**: 实时过滤当前加载的数据。
    *   **列筛选**: 自动提取某列唯一值，支持精确过滤。
    *   **列可见性**: 随时隐藏/显示特定列，专注关键数据。
*   **JSON 友好**:
    *   自动识别单元格内的 JSON 字符串。
    *   提供**格式化**和**可折叠**的树状视图，告别黑压压的字符串堆砌。
*   **图像预览**: 自动识别图片 URL 并直接渲染预览图。

## 快速开始 (Quick Start)

本项目使用 `uv` 进行现代化的 Python 依赖管理。

### 1. 安装环境
```bash
# 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并进入目录
git clone <repo_url>
cd fserver

# 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
uv pip install -r requirements.txt
```

### 2. 启动服务
```bash
# 开发模式启动 (支持热重载)
uvicorn fserver:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/list/` 开始使用。

## API 指南 (API Guide)

### 浏览与查看
*   `GET /list/{path}`: 浏览目录或跳转到文件查看器。
*   `GET /tsv/{path}`: 表格数据查看器（HTML页面）。
    *   `start`: 起始行 (默认 0)。
    *   `length`: 加载条数 (默认 1000)。
    *   `key` / `value`: 按列名和值过滤。
    *   `json_cols`: 指定需要 JSON 格式化的列（逗号分隔）。
    *   `hide_cols`: 指定默认隐藏的列。

### 数据导出
*   `GET /excel/{path}`: 将指定文件转换为 Excel 下载。
*   `GET /download/{path}`: 下载原始文件。
*   `GET /txt/{path}`: 以纯文本形式查看文件。

### 工具接口
*   `GET /api/tsv/key/{path}`: 获取某列的所有唯一值（用于前端下拉框）。
*   `GET /json_viewer`: 独立的 JSON 美化查看页面。

## 架构说明 (Architecture)

*   **Backend**: Python 3.12+, FastAPI, Pandas (核心数据处理), Aiofiles (异步IO).
*   **Caching**: 使用 `cachetools` 在内存中缓存 DataFrame，避免重复读取磁盘，大幅提升翻页和过滤速度。
*   **Frontend**: Jinja2 模板渲染 HTML，jQuery DataTables 负责交互。利用 `dt-colresize` 和 `buttons` 扩展增强体验。

## 待办事项 (Todo)

- [ ] **性能**: 大文件加载优化（目前全量加载进内存，通过切片分页，未来计划支持流式读取）。
- [ ] **功能**: 简单的文本文件在线编辑。
- [ ] **部署**: Docker 镜像支持。

