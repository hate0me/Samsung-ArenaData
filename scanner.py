#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import docx
import pandas as pd
import pypdf
import pytesseract
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from striprtf.striprtf import rtf_to_text
from tqdm import tqdm

import warnings
import logging

# Отключаем мусорные ворнинги от библиотек
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pypdf").setLevel(logging.CRITICAL)
# ==========================================
# 1. КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ==========================================
LARGE_VOLUME_THRESHOLD = 100
MAX_TEXT_LENGTH = 300_000  # Ограничение текста для ускорения NLP
DEFAULT_OCR_LANG = "rus+eng"  # Можно добавить ara, nld и т.д.

CAT_SPECIAL = {'biometric', 'health', 'religion', 'race', 'political'}
CAT_GOV_ID = {'snils', 'inn', 'passport', 'mrz'}
CAT_PAYMENT = {'credit_card', 'bank_account', 'bik'}
CAT_COMMON = {'full_name', 'phone', 'email', 'birth_date', 'address'}

# Глобальная переменная для воркеров (чтобы не копировать тяжелую модель в памяти)
_analyzer = None


# ==========================================
# 2. ВАЛИДАТОРЫ ПДн
# ==========================================
def validate_snils(number_str: str) -> bool:
    digits = re.sub(r'\D', '', number_str)
    if len(digits) != 11 or digits == "00000000000": return False
    total = sum(int(digits[i]) * (9 - i) for i in range(9))
    check = total % 101 if total % 101 != 100 else 0
    return check == int(digits[9:])


def validate_inn(inn_str: str) -> bool:
    digits = re.sub(r'\D', '', inn_str)
    if len(digits) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return (sum(coeffs[i] * int(digits[i]) for i in range(9)) % 11 % 10) == int(digits[9])
    elif len(digits) == 12:
        coeffs10 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs11 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        check10 = sum(coeffs10[i] * int(digits[i]) for i in range(10)) % 11 % 10
        check11 = sum(coeffs11[i] * int(digits[i]) for i in range(11)) % 11 % 10
        return check10 == int(digits[10]) and check11 == int(digits[11])
    return False


def luhn_check(card_number: str) -> bool:
    digits = re.sub(r'\D', '', card_number)
    if not (13 <= len(digits) <= 19): return False
    total = sum(int(d) if i % 2 == 0 else (int(d) * 2 - 9 if int(d) * 2 > 9 else int(d) * 2)
                for i, d in enumerate(digits[::-1]))
    return total % 10 == 0


# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ PRESIDIO (NLP)
# ==========================================
def create_presidio_analyzer() -> AnalyzerEngine:
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["ru", "en"])

    # Кастомные паттерны
    patterns = {
        'SNILS': r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b",
        'INN': r"\b\d{10}\b|\b\d{12}\b",
        'PASSPORT_RF': r"\b\d{4}[-\s]?\d{6}\b",
        'CREDIT_CARD': r"\b(?:\d[ -]*?){13,19}\b",
        'MRZ': r"\b[A-Z0-9<]{30,44}\n[A-Z0-9<]{30,44}\b"
    }
    for entity, pattern in patterns.items():
        rec = PatternRecognizer(supported_entity=entity,
                                patterns=[Pattern(name=entity.lower(), regex=pattern, score=0.8)],
                                supported_language="ru")
        analyzer.registry.add_recognizer(rec)

    keywords = {
        'BIOMETRIC_KEYWORD': ["биометрия", "отпечаток пальца", "радужка", "голосовой образец"],
        'HEALTH_KEYWORD': ["диагноз", "медицинская карта", "история болезни", "инвалидность"],
        'RELIGION_KEYWORD': ["вероисповедание", "религия", "православие", "ислам"],
    }
    for entity, kw_list in keywords.items():
        pattern = r"\b(" + "|".join(kw_list) + r")\b"
        rec = PatternRecognizer(supported_entity=entity,
                                patterns=[Pattern(name=entity.lower(), regex=pattern, score=0.9)],
                                supported_language="ru")
        analyzer.registry.add_recognizer(rec)

    return analyzer


def init_worker():
    """Инициализация ресурсов внутри каждого процесса-воркера"""
    global _analyzer
    _analyzer = create_presidio_analyzer()


# ==========================================
# 4. ИЗВЛЕЧЕНИЕ ТЕКСТА И OCR (TESSERACT)
# ==========================================
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:MAX_TEXT_LENGTH]


def ocr_image_pil(img: Image.Image, lang: str) -> str:
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = gray.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(gray, lang=lang)


def extract_text(file_path: Path, ocr_lang: str) -> str:
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    declared_ext = file_path.suffix.lower()

    if declared_ext in ['.mp4', '.avi', '.mov', '.mkv'] and file_size_mb > 500:
        return ""
    if declared_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'] and file_size_mb > 50:
        return ""

    text = ""
    try:
        # Умное определение типа файла по первым байтам
        with open(file_path, 'rb') as f:
            header = f.read(20).lower()

        real_ext = declared_ext
        if b'<!doc' in header or b'<html' in header:
            real_ext = '.html'
        elif b'%pdf' in header:
            real_ext = '.pdf'
        elif header.startswith(b'{') or header.startswith(b'['):
            real_ext = '.json'

        # Теперь используем реальное расширение
        if real_ext in ['.csv', '.json', '.parquet', '.xls', '.xlsx']:
            if real_ext == '.csv':
                df = pd.read_csv(file_path, nrows=5000, encoding_errors='ignore')
            elif real_ext == '.json':
                df = pd.read_json(file_path)
            elif real_ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_excel(file_path)
            text = ' '.join(df.astype(str).values.flatten())

        elif real_ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = '\n'.join([p.extract_text() for p in reader.pages[:30] if p.extract_text()])

        elif real_ext == '.docx':
            doc = docx.Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs])

        elif real_ext == '.rtf':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = rtf_to_text(f.read(MAX_TEXT_LENGTH))

        elif real_ext in ['.html', '.htm']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(MAX_TEXT_LENGTH), 'html.parser')
                text = soup.get_text(separator=' ')

        elif real_ext in ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp']:
            img = Image.open(file_path)
            text = ocr_image_pil(img, ocr_lang)

        elif real_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                step = max(1, int(fps * 3.0))  # Берем кадр раз в 3 секунды для скорости
                idx = 0
                frames_text = []
                while True:
                    ok, frame = cap.read()
                    if not ok or len(frames_text) > 5: break  # Максимум 5 кадров с видео
                    if idx % step == 0:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frames_text.append(ocr_image_pil(img, ocr_lang))
                    idx += 1
                text = "\n".join(frames_text)
                cap.release()

        else:  # Fallback для обычных текстовых файлов
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(MAX_TEXT_LENGTH)

    except Exception as e:
        # Теперь ошибки будут тихо падать в дебаг, не останавливая процесс
        logging.debug(f"Пропущен файл {file_path.name} из-за ошибки: {e}")

    return normalize_text(text)


# ==========================================
# 5. АНАЛИЗ ПДн И УЗ
# ==========================================
def analyze_text(text: str) -> Dict[str, int]:
    if not text: return {}

    results = _analyzer.analyze(text=text, language='ru', entities=[])
    counts = defaultdict(int)
    seen = defaultdict(set)

    mapping = {
        'PERSON': 'full_name', 'PHONE_NUMBER': 'phone', 'EMAIL_ADDRESS': 'email',
        'DATE_TIME': 'birth_date', 'ADDRESS': 'address', 'SNILS': 'snils',
        'INN': 'inn', 'PASSPORT_RF': 'passport', 'CREDIT_CARD': 'credit_card',
        'BANK_ACCOUNT': 'bank_account', 'MRZ': 'mrz', 'BIOMETRIC_KEYWORD': 'biometric',
        'HEALTH_KEYWORD': 'health', 'RELIGION_KEYWORD': 'religion'
    }

    for res in results:
        val = text[res.start:res.end]
        cat = mapping.get(res.entity_type, res.entity_type.lower())

        valid = True
        if res.entity_type == 'SNILS':
            valid = validate_snils(val)
        elif res.entity_type == 'INN':
            valid = validate_inn(val)
        elif res.entity_type == 'CREDIT_CARD':
            valid = luhn_check(val)

        if valid and val not in seen[cat]:
            seen[cat].add(val)
            counts[cat] += 1

    return dict(counts)


def classify_uz(counts: Dict[str, int]) -> str:
    if any(c in CAT_SPECIAL for c in counts): return "УЗ-1"

    pay_count = sum(counts.get(c, 0) for c in CAT_PAYMENT)
    gov_count = sum(counts.get(c, 0) for c in CAT_GOV_ID)
    com_count = sum(counts.get(c, 0) for c in CAT_COMMON)

    if pay_count > 0 or gov_count > LARGE_VOLUME_THRESHOLD: return "УЗ-2"
    if (0 < gov_count <= LARGE_VOLUME_THRESHOLD) or com_count > LARGE_VOLUME_THRESHOLD: return "УЗ-3"
    if com_count > 0: return "УЗ-4"
    return "Не определен"


# ==========================================
# 6. ОРКЕСТРАЦИЯ
# ==========================================
def process_file_worker(file_path: Path, ocr_lang: str) -> Optional[Dict]:
    text = extract_text(file_path, ocr_lang)
    counts = analyze_text(text)

    if not counts:
        return None

    return {
        'path': str(file_path),
        'categories': ', '.join(counts.keys()),
        'total_occurrences': sum(counts.values()),
        'security_level': classify_uz(counts),
        'file_format': file_path.suffix.lower()[1:] or 'unknown'
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Папка для сканирования (напр. /DATA)")
    p.add_argument("-o", "--output", default="report.csv", help="Путь к отчету (CSV или JSON)")
    p.add_argument("-l", "--lang", default=DEFAULT_OCR_LANG, help="Языки Tesseract (напр. rus+eng+ara)")
    p.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4, help="Кол-во процессов")
    args = p.parse_args()

    root = Path(args.input).resolve()
    if not root.exists():
        print(f"Ошибка: Директория {root} не найдена.")
        return

    files = [p for p in root.rglob('*') if p.is_file() and p.name != args.output]
    print(f"🚀 Найдено файлов: {len(files)}. Запуск {args.workers} процессов...")

    results = []
    # Используем ProcessPoolExecutor для максимальной утилизации CPU
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_file_worker, f, args.lang): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Сканирование"):
            try:
                res = future.result()
                if res: results.append(res)
            except Exception as e:
                logging.debug(f"Process crashed on file: {e}")

    # Сохранение отчета
    out_path = Path(args.output)
    if out_path.suffix.lower() == '.json':
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'categories', 'total_occurrences', 'security_level',
                                                   'file_format'])
            writer.writeheader()
            writer.writerows(results)

    print(f"✅ Готово! Файлов с ПДн: {len(results)}. Отчет сохранен в {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()