"""Парсинг банковских выписок: Kaspi Gold PDF (текстовый слой) и универсальный CSV."""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime

# Строка выписки Kaspi Gold, пример:
#   "01.11.24 - 2 500,00 ₸ Покупки Magnum Cash&Carry"
#   "03.11.24 + 150 000,00 ₸ Пополнения С Kaspi Депозита"
KASPI_LINE = re.compile(
    r"^(?P<date>\d{2}\.\d{2}\.\d{2,4})\s+(?P<sign>[+-])\s*(?P<amount>[\d\s ]+(?:[.,]\d{1,2})?)\s*₸\s+"
    r"(?P<type>\S+)\s*(?P<desc>.*)$"
)

DATE_FORMATS = ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d")


class ParseError(ValueError):
    pass


def parse_date(s: str) -> date:
    s = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ParseError(f"Не удалось распознать дату: {s!r}")


def parse_amount(s: str) -> float:
    s = str(s).replace("₸", "").replace(" ", "").replace(" ", "").strip()
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    return float(s)


def parse_kaspi_text(text: str) -> list[dict]:
    rows = []
    for raw in text.splitlines():
        m = KASPI_LINE.match(raw.strip())
        if not m:
            continue
        amount = parse_amount(m["amount"])
        if m["sign"] == "-":
            amount = -amount
        rows.append({
            "date": parse_date(m["date"]).isoformat(),
            "amount": amount,
            "op_type": m["type"],
            "description": m["desc"].strip() or m["type"],
        })
    return rows


def parse_pdf(data: bytes) -> list[dict]:
    import pdfplumber  # импорт по требованию — тяжёлая зависимость

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    rows = parse_kaspi_text(text)
    if not rows:
        raise ParseError("В PDF не найдено операций. Поддерживается выписка Kaspi Gold (не скан).")
    return rows


# Синонимы колонок CSV разных банков
COLS = {
    "date": ("date", "дата", "дата операции", "transaction date"),
    "amount": ("amount", "сумма", "сумма операции", "sum"),
    "description": ("description", "описание", "детали", "назначение", "merchant", "мерчант"),
    "op_type": ("type", "операция", "тип", "op_type"),
}


def parse_csv(data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ParseError("Пустой CSV")
    header = {h.strip().lower(): h for h in reader.fieldnames}
    mapping = {}
    for key, syns in COLS.items():
        for s in syns:
            if s in header:
                mapping[key] = header[s]
                break
    missing = {"date", "amount", "description"} - mapping.keys()
    if missing:
        raise ParseError(f"В CSV нет колонок: {', '.join(sorted(missing))}")
    rows = []
    for r in reader:
        try:
            rows.append({
                "date": parse_date(r[mapping["date"]]).isoformat(),
                "amount": parse_amount(r[mapping["amount"]]),
                "op_type": (r.get(mapping["op_type"]) or "").strip() if "op_type" in mapping else "",
                "description": (r[mapping["description"]] or "").strip(),
            })
        except (ParseError, ValueError):
            continue  # битые строки пропускаем
    if not rows:
        raise ParseError("Не найдено ни одной корректной строки")
    return rows


def parse_file(filename: str, data: bytes) -> list[dict]:
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        return parse_pdf(data)
    if name.endswith((".csv", ".txt")):
        return parse_csv(data)
    raise ParseError("Поддерживаются файлы .pdf (Kaspi) и .csv")
