"""Генерирует реалистичную демо-выписку за 4 месяца (CSV) и Kaspi-подобный текст."""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

MERCHANTS = [  # (описание, тип, мин, макс, вероятность в день)
    ("Magnum Cash&Carry", "Покупки", 3000, 25000, 0.35),
    ("Small", "Покупки", 500, 4000, 0.30),
    ("Yandex Go", "Покупки", 900, 3500, 0.45),
    ("Wolt", "Покупки", 3500, 9000, 0.15),
    ("Glovo", "Покупки", 3000, 8000, 0.08),
    ("Coffee Boom", "Покупки", 1200, 2500, 0.25),
    ("Salam Bro", "Покупки", 2500, 6000, 0.08),
    ("Аптека Europharma", "Покупки", 1500, 12000, 0.05),
    ("Kaspi Магазин", "Покупки", 5000, 60000, 0.05),
    ("Wildberries", "Покупки", 3000, 20000, 0.06),
    ("Helios AZS", "Покупки", 8000, 18000, 0.07),
    ("Kinopark", "Покупки", 2500, 7000, 0.04),
    ("Zara", "Покупки", 15000, 45000, 0.02),
    ("Перевод Айгерим С.", "Переводы", 2000, 30000, 0.08),
]
MONTHLY = [  # (день, описание, тип, сумма)
    (5, "Netflix.com", "Покупки", 3790), (7, "Spotify AB", "Покупки", 1690),
    (9, "Beeline KZ", "Покупки", 4500), (12, "ChatGPT OPENAI", "Покупки", 10500),
    (15, "Алсеко", "Покупки", 18500), (20, "Fitness Invictus", "Покупки", 25000),
    (10, "Зарплата ТОО Компания", "Пополнения", 450000), (25, "Аванс ТОО Компания", "Пополнения", 200000),
]


def generate(start: date = date(2026, 7, 1), days: int = 123, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        growth = 1 + 0.5 * (i / days)  # доставка постепенно растёт — для инсайта
        for desc, typ, lo, hi, p in MERCHANTS:
            pp = p * (growth if desc in ("Wolt", "Glovo") else 1)
            if rnd.random() < pp:
                rows.append({"date": d, "amount": -round(rnd.uniform(lo, hi), -1), "type": typ, "description": desc})
        for day, desc, typ, amt in MONTHLY:
            if d.day == day:
                sign = 1 if typ == "Пополнения" else -1
                rows.append({"date": d, "amount": sign * amt, "type": typ, "description": desc})
    rows.append({"date": start + timedelta(days=70), "amount": -189000, "type": "Покупки",
                 "description": "Аптека Europharma"})  # аномалия
    return sorted(rows, key=lambda r: r["date"])


def write_demo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Дата", "Сумма", "Операция", "Детали"])
        for r in generate():
            w.writerow([r["date"].strftime("%d.%m.%Y"), r["amount"], r["type"], r["description"]])
    return path


def kaspi_text(rows: list[dict]) -> str:
    lines = ["Kaspi Gold Выписка", "Дата Сумма Операция Детали"]
    for r in rows:
        sign = "+" if r["amount"] > 0 else "-"
        amt = f"{abs(r['amount']):,.2f}".replace(",", " ").replace(".", ",")
        lines.append(f"{r['date']:%d.%m.%y} {sign} {amt} ₸ {r['type']} {r['description']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(write_demo(Path(__file__).parent / "demo_statement.csv"))
