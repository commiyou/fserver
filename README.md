# fserver

基于python3 fastapi +jinja2 + jquery + datatables 的开发机文件服务器，可以作为资源管理器 上传&下载文件、查看tsv

## INSTALL
```shell
pip3 install -r ./requirements.txt
```

## RUN

api doc:
- http://127.0.0.1:8113/docs
- http://127.0.0.1:8113/redoc

```shell
uvicorn fserver:app --reload --host 0.0.0.0 --port 8113
```
