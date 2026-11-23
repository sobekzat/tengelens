"""ML-категоризатор транзакций.

Гибрид: сначала точные правила по типу операции (переводы, пополнения, снятия),
затем ML-модель (TF-IDF по символьным n-граммам + Logistic Regression),
обученная на размеченных названиях мерчантов Казахстана. Символьные n-граммы
устойчивы к опечаткам, транслиту и разным написаниям ("MAGNUM", "Магнум", "Magnum Cash&Carry").
"""
from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = [
    "Продукты", "Кафе и рестораны", "Такси и транспорт", "Доставка еды",
    "Подписки", "Связь и интернет", "Коммунальные", "Одежда и обувь",
    "Здоровье и аптеки", "Развлечения", "Маркетплейсы", "Образование",
    "Красота", "Авто и топливо", "Переводы", "Пополнения", "Снятие наличных", "Другое",
]

# Размеченная обучающая выборка: (текст мерчанта, категория)
SEED: dict[str, list[str]] = {
    "Продукты": ["Magnum Cash&Carry", "Магнум", "MAGNUM SUPERMARKET", "Small", "SMALL магазин",
                 "Galmart", "Ramstore", "Метро Кэш энд Керри", "METRO", "Анвар", "ANVAR",
                 "Toimart", "Arzan", "Супермаркет Дина", "Продуктовый магазин", "Minimarket 24",
                 "Fix Price", "Daramart", "Interfood", "Green Market", "Universam"],
    "Кафе и рестораны": ["Starbucks", "Coffee Boom", "Del Papa", "Kofeinya", "Burger King",
                         "KFC", "McDonalds", "Salam Bro", "Pizza Hut", "Dodo Pizza", "Navat",
                         "Cafe Central", "Ресторан Алаша", "Кофейня Zebra", "Tomas Coffee", "Bahandi",
                         "Paul Bakery", "Doner Kebab", "Шашлычная", "Sushi Master"],
    "Такси и транспорт": ["Yandex Go", "YANDEX.TAXI", "Яндекс Такси", "inDrive", "INDRIVER",
                          "Uber", "Onay", "ONAY ALMATY", "Автобус", "Metro Almaty", "Bolt",
                          "Air Astana", "FlyArystan", "КТЖ", "Temir Zholy", "Aviata"],
    "Доставка еды": ["Wolt", "WOLT KZ", "Glovo", "GLOVO DELIVERY", "Chocofood", "Yandex Eda",
                     "Яндекс Еда", "Arbuz.kz", "ARBUZ", "Choco Delivery", "Yandex Lavka"],
    "Подписки": ["Netflix", "NETFLIX.COM", "Spotify", "SPOTIFY AB", "Apple.com/bill", "APPLE COM BILL",
                 "Google Play", "GOOGLE *YouTube", "YouTube Premium", "Yandex Plus", "Яндекс Плюс",
                 "ChatGPT", "OPENAI", "Claude.ai", "ANTHROPIC", "iCloud", "Kinopoisk", "Ivi",
                 "Telegram Premium", "Microsoft 365", "Adobe", "Notion", "Steam subscription"],
    "Связь и интернет": ["Beeline", "BEELINE KZ", "Kcell", "KCELL", "Activ", "Tele2", "TELE2 KZ",
                         "Altel", "Kazakhtelecom", "Казахтелеком", "IDNet", "Megaline", "Билайн"],
    "Коммунальные": ["Алматы Энергосбыт", "AlmatyEnergoSbyt", "Алсеко", "ALSECO", "Астана ЕРЦ",
                     "Водоканал", "КазТрансГаз", "QazaqGaz", "ОСИ", "Тепловые сети", "Коммунальные услуги",
                     "ЖКХ оплата", "ЕРЦ"],
    "Одежда и обувь": ["Zara", "ZARA", "H&M", "LC Waikiki", "Bershka", "Pull&Bear", "Adidas",
                       "Nike", "Puma", "Colin's", "Mango", "Massimo Dutti", "Stradivarius", "Ecco",
                       "Lamoda", "Gloria Jeans", "Defacto", "Koton"],
    "Здоровье и аптеки": ["Аптека Europharma", "EUROPHARMA", "Biosfera", "Аптека Plus", "Sadykhan",
                          "Медицинский центр", "Клиника Invivo", "INVITRO", "Олимп лаборатория",
                          "Стоматология", "Аптека 36.6", "Медикер", "Dent", "Стомат", "Green Clinic"],
    "Развлечения": ["Kinopark", "KINOPARK", "Chaplin Cinema", "Steam", "STEAM GAMES", "PlayStation",
                    "Боулинг", "Ticketon", "TICKETON", "Kassir.kz", "Шымбулак", "Shymbulak",
                    "Медеу", "Аквапарк", "Квест", "Fitness Invictus", "Invictus", "World Class"],
    "Маркетплейсы": ["Kaspi Магазин", "Kaspi.kz магазин", "Wildberries", "WILDBERRIES", "Ozon",
                     "OZON", "AliExpress", "ALIEXPRESS", "Temu", "TEMU", "Amazon", "Halyk Market",
                     "Lamoda Marketplace", "Flip.kz", "Sulpak", "Technodom", "Mechta", "Alser"],
    "Образование": ["Coursera", "COURSERA", "Udemy", "UDEMY", "Stepik", "Skillbox", "Яндекс Практикум",
                    "Duolingo", "Меломан книги", "Marwin", "Университет оплата", "SDU University",
                    "Курсы английского", "IELTS registration", "LeetCode", "DataCamp"],
    "Красота": ["Barbershop", "Барбершоп", "Салон красоты", "Золотое Яблоко", "Летуаль",
                "Sephora", "Beauty salon", "Маникюр", "Spa", "Cosmetics shop"],
    "Авто и топливо": ["Helios", "HELIOS AZS", "Sinooil", "SINOOIL", "Qazaq Oil", "КазМунайГаз",
                       "Gazprom AZS", "Royal Petrol", "Автомойка", "Шиномонтаж", "Parking Almaty",
                       "Паркинг", "СТО", "Lukoil", "Компас АЗС"],
}

# Правила по типу операции из выписки (точные, имеют приоритет над ML)
TYPE_RULES = {
    "перевод": "Переводы", "переводы": "Переводы",
    "пополнение": "Пополнения", "пополнения": "Пополнения",
    "снятие": "Снятие наличных", "снятия": "Снятие наличных",
}
KEYWORD_RULES = [
    (re.compile(r"\b(перевод|transfer|p2p)\b", re.I), "Переводы"),
    (re.compile(r"\b(банкомат|atm|снятие)\b", re.I), "Снятие наличных"),
    (re.compile(r"\b(пополнение|зарплата|salary|cashback|кешбэк)\b", re.I), "Пополнения"),
]

MIN_CONFIDENCE = 0.18  # ниже порога — "Другое"


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[*#№]+", " ", text)
    text = re.sub(r"\d{3,}", " ", text)  # номера терминалов/чеков — шум
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    texts, labels = [], []
    for cat, names in SEED.items():
        for n in names:
            # аугментация: варианты регистра и типичные приписки из выписок
            for variant in (n, n.upper(), f"{n} Almaty", f"Покупка {n}", f"{n} KZ"):
                texts.append(_normalize(variant))
                labels.append(cat)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, C=8.0)),
    ])
    model.fit(texts, labels)
    return model


def categorize(description: str, op_type: str = "", amount: float = 0.0) -> tuple[str, float]:
    """Возвращает (категория, уверенность 0..1)."""
    t = (op_type or "").strip().lower()
    if t in TYPE_RULES:
        return TYPE_RULES[t], 1.0
    for rx, cat in KEYWORD_RULES:
        if rx.search(description or ""):
            return cat, 0.95
    if amount > 0:
        return "Пополнения", 0.9
    model = get_model()
    proba = model.predict_proba([_normalize(description or "")])[0]
    i = int(proba.argmax())
    conf = float(proba[i])
    if conf < MIN_CONFIDENCE:
        return "Другое", conf
    return str(model.classes_[i]), conf


def categorize_many(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        cat, conf = categorize(r.get("description", ""), r.get("op_type", ""), r.get("amount", 0.0))
        out.append({**r, "category": cat, "confidence": round(conf, 3)})
    return out
