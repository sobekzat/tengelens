"""TengeLens — FastAPI-приложение: загрузка выписки → категоризация → аналитика."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analytics import analyze
from .categorizer import categorize_many
from .parser import ParseError, parse_file

BASE = Path(__file__).resolve().parent.parent
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="TengeLens", version="1.0.0",
              description="Анализ банковских выписок Kaspi/CSV с ML-категоризацией расходов")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Файл больше 10 МБ")
    try:
        rows = parse_file(file.filename or "", data)
    except ParseError as e:
        raise HTTPException(422, str(e))
    # Файл не сохраняется: обработка только в памяти (приватность — ключевое обещание продукта)
    return analyze(categorize_many(rows))


@app.get("/api/demo")
def demo():
    path = BASE / "sample" / "demo_statement.csv"
    if not path.exists():
        from sample.generate import write_demo  # type: ignore
        write_demo(path)
    rows = parse_file(path.name, path.read_bytes())
    return analyze(categorize_many(rows))
