"""Аналитика расходов: сводки, подписки, аномалии, прогноз и персональные инсайты."""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

NON_SPENDING = {"Переводы", "Пополнения"}


def _merchant_key(desc: str) -> str:
    s = re.sub(r"[^a-zа-яё ]", " ", desc.lower())
    words = [w for w in s.split() if w not in {"покупка", "оплата", "kz", "almaty", "астана", "алматы"}]
    return " ".join(words[:2]) or desc.lower()


def build_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["merchant"] = df["description"].map(_merchant_key)
    df["spend"] = np.where((df["amount"] < 0) & (~df["category"].isin(NON_SPENDING)), -df["amount"], 0.0)
    return df.sort_values("date").reset_index(drop=True)


def detect_subscriptions(df: pd.DataFrame) -> list[dict]:
    """Регулярный платёж = один мерчант, ≥2 списания, интервал 25–35 дней, сумма стабильна (±15%)."""
    subs = []
    out = df[df["amount"] < 0]
    for merchant, g in out.groupby("merchant"):
        if len(g) < 2:
            continue
        gaps = g["date"].diff().dt.days.dropna()
        amounts = -g["amount"]
        if gaps.between(25, 35).mean() >= 0.7 and amounts.std(ddof=0) <= 0.15 * amounts.mean():
            subs.append({
                "key": merchant,
                "merchant": g["description"].iloc[-1],
                "monthly": round(float(amounts.mean()), 0),
                "yearly": round(float(amounts.mean()) * 12, 0),
                "count": int(len(g)),
                "last_date": g["date"].iloc[-1].date().isoformat(),
                "category": g["category"].mode().iloc[0],
            })
    return sorted(subs, key=lambda s: -s["monthly"])


def detect_anomalies(df: pd.DataFrame, z: float = 3.5, recurring: set[str] | None = None) -> list[dict]:
    """Траты, аномально крупные для своей категории: робастный z-score (MAD) и > 5× медианы категории.
    Регулярные платежи (подписки) исключаются — они ожидаемы."""
    res = []
    sp = df[(df["spend"] > 0) & (~df["merchant"].isin(recurring or set()))].copy()
    # База сравнения: сам мерчант (если у него ≥5 покупок), иначе категория
    counts = sp["merchant"].map(sp["merchant"].value_counts())
    sp["base"] = np.where(counts >= 5, "m:" + sp["merchant"], "c:" + sp["category"])
    for _, g in sp.groupby("base"):
        if len(g) < 5:
            continue
        med = g["spend"].median()
        mad = (g["spend"] - med).abs().median() or 1.0
        score = 0.6745 * (g["spend"] - med) / mad
        for _, r in g[(score > z) & (g["spend"] > 5 * med)].iterrows():
            res.append({"date": r["date"].date().isoformat(), "description": r["description"],
                        "amount": round(float(r["spend"]), 0), "category": r["category"],
                        "typical": round(float(med), 0)})
    return sorted(res, key=lambda x: -x["amount"])[:10]


def forecast_next_month(df: pd.DataFrame) -> dict:
    """Прогноз трат на следующий месяц по категориям: взвешенное среднее последних 3 полных месяцев."""
    monthly = df.pivot_table(index="month", columns="category", values="spend", aggfunc="sum").fillna(0)
    if monthly.empty:
        return {"total": 0, "by_category": {}}
    last_day = df["date"].max()
    if last_day.day < last_day.days_in_month - 2 and len(monthly) > 1:
        monthly = monthly.iloc[:-1]  # последний месяц неполный — не учитываем
    tail = monthly.tail(3)
    weights = np.arange(1, len(tail) + 1, dtype=float)
    weights /= weights.sum()
    pred = (tail.mul(weights, axis=0)).sum()
    pred = pred[pred > 0].sort_values(ascending=False)
    return {"total": round(float(pred.sum()), 0),
            "by_category": {k: round(float(v), 0) for k, v in pred.items()}}


def insights(df: pd.DataFrame, subs: list[dict]) -> list[str]:
    tips = []
    monthly = df.pivot_table(index="month", columns="category", values="spend", aggfunc="sum").fillna(0)
    if len(monthly) >= 2:
        prev, cur = monthly.iloc[-2], monthly.iloc[-1]
        for cat in cur.index:
            if prev[cat] > 5000 and cur[cat] > prev[cat] * 1.3 and cur[cat] - prev[cat] > 15000:
                tips.append(f"«{cat}»: рост на {int((cur[cat] / prev[cat] - 1) * 100)}% к прошлому месяцу "
                            f"({int(prev[cat]):,} → {int(cur[cat]):,} ₸).".replace(",", " "))
    if subs:
        total = sum(s["yearly"] for s in subs)
        tips.append(f"Найдено {len(subs)} регулярных платежей на {int(total):,} ₸ в год. "
                    f"Отмена ненужных — самая быстрая экономия.".replace(",", " "))
    sp = df[df["spend"] > 0]
    if not sp.empty:
        delivery = sp.loc[sp["category"] == "Доставка еды", "spend"].sum()
        food = sp.loc[sp["category"] == "Продукты", "spend"].sum()
        if delivery > 0.5 * max(food, 1):
            tips.append(f"На доставку уходит {int(delivery / max(food, 1) * 100)}% от суммы на продукты. "
                        f"Готовка дома 3 раза в неделю сэкономит ≈{int(delivery * 0.4):,} ₸ за период.".replace(",", " "))
        small = sp[sp["spend"] < 2000]
        if len(small) > 20:
            tips.append(f"{len(small)} мелких покупок до 2 000 ₸ дали в сумме {int(small['spend'].sum()):,} ₸ — "
                        f"«эффект латте».".replace(",", " "))
    income = df.loc[df["category"] == "Пополнения", "amount"].sum()
    spend = sp["spend"].sum() if not sp.empty else 0
    if income > 0:
        rate = 1 - spend / income
        tips.append(f"Норма сбережений: {int(rate * 100)}%. "
                    + ("Отлично — цель 20%+ достигнута." if rate >= 0.2 else "Рекомендуемая цель — 20% от дохода."))
    return tips


def analyze(rows: list[dict]) -> dict:
    df = build_frame(rows)
    sp = df[df["spend"] > 0]
    by_cat = sp.groupby("category")["spend"].sum().sort_values(ascending=False)
    by_month = df.groupby("month").agg(spend=("spend", "sum"),
                                       income=("amount", lambda s: s[s > 0].sum())).reset_index()
    top_merchants = (sp.groupby("description")["spend"].agg(["sum", "count"])
                     .sort_values("sum", ascending=False).head(10))
    cat_month = defaultdict(dict)
    for (m, c), v in sp.groupby(["month", "category"])["spend"].sum().items():
        cat_month[c][m] = round(float(v), 0)
    subs = detect_subscriptions(df)
    return {
        "period": {"from": df["date"].min().date().isoformat(), "to": df["date"].max().date().isoformat()},
        "totals": {
            "spend": round(float(sp["spend"].sum()), 0),
            "income": round(float(df.loc[df["amount"] > 0, "amount"].sum()), 0),
            "transactions": int(len(df)),
        },
        "by_category": {k: round(float(v), 0) for k, v in by_cat.items()},
        "by_month": by_month.round(0).to_dict(orient="records"),
        "category_by_month": cat_month,
        "top_merchants": [{"merchant": k, "sum": round(float(r["sum"]), 0), "count": int(r["count"])}
                          for k, r in top_merchants.iterrows()],
        "subscriptions": subs,
        "anomalies": detect_anomalies(df, recurring={s["key"] for s in subs}),
        "forecast": forecast_next_month(df),
        "insights": insights(df, subs),
        "transactions": [
            {"date": r["date"].date().isoformat(), "description": r["description"], "amount": r["amount"],
             "category": r["category"], "confidence": r["confidence"]}
            for _, r in df.sort_values("date", ascending=False).iterrows()
        ],
    }
