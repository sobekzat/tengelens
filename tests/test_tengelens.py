from pathlib import Path

from fastapi.testclient import TestClient

from app.analytics import analyze
from app.categorizer import categorize, categorize_many
from app.main import app
from app.parser import parse_csv, parse_kaspi_text
from sample.generate import generate, kaspi_text

client = TestClient(app)


def test_categorizer_known_merchants():
    assert categorize("MAGNUM SUPERMARKET")[0] == "Продукты"
    assert categorize("YANDEX.TAXI")[0] == "Такси и транспорт"
    assert categorize("NETFLIX.COM 4029357733")[0] == "Подписки"
    assert categorize("Wolt KZ")[0] == "Доставка еды"
    assert categorize("Любой текст", op_type="Переводы")[0] == "Переводы"
    assert categorize("Зарплата", amount=100000)[0] == "Пополнения"


def test_categorizer_unseen_spelling():
    # написания, которых нет в обучающей выборке
    assert categorize("Magnum Express Astana")[0] == "Продукты"
    assert categorize("Glovo App")[0] == "Доставка еды"


def test_parse_kaspi_text():
    text = ("Kaspi Gold\n01.11.24 - 2 500,00 ₸ Покупки Magnum Cash&Carry\n"
            "03.11.24 + 150 000,00 ₸ Пополнения С Kaspi Депозита\nмусорная строка")
    rows = parse_kaspi_text(text)
    assert len(rows) == 2
    assert rows[0] == {"date": "2024-11-01", "amount": -2500.0, "op_type": "Покупки",
                       "description": "Magnum Cash&Carry"}
    assert rows[1]["amount"] == 150000.0


def test_parse_csv_semicolon_and_synonyms():
    data = "Дата;Сумма;Описание\n01.10.2026;-1 200,50;Starbucks\n02.10.2026;bad;X\n".encode()
    rows = parse_csv(data)
    assert len(rows) == 1 and rows[0]["amount"] == -1200.5


def test_full_analysis_on_generated_kaspi_statement():
    rows = parse_kaspi_text(kaspi_text(generate()))
    report = analyze(categorize_many(rows))
    subs = {s["merchant"] for s in report["subscriptions"]}
    assert {"Netflix.com", "Spotify AB", "ChatGPT OPENAI"} <= subs
    assert any(a["amount"] == 189000 for a in report["anomalies"])
    assert report["forecast"]["total"] > 0
    assert report["totals"]["income"] == 2_600_000


def test_api_demo_and_upload():
    r = client.get("/api/demo")
    assert r.status_code == 200 and r.json()["totals"]["transactions"] > 100
    csv_bytes = Path("sample/demo_statement.csv").read_bytes()
    r = client.post("/api/analyze", files={"file": ("st.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    r = client.post("/api/analyze", files={"file": ("x.docx", b"hello", "application/octet-stream")})
    assert r.status_code == 422
