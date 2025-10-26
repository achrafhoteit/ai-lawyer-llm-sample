# llm_extract.py
import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import boto3

# -----------------------------
# Config / Constants
# -----------------------------
DEFAULT_REGION = "us-east-1"
# Bedrock model IDs (streaming Claude 3.7 Sonnet)
DEFAULT_STREAM_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# Non-streaming (short classification), keep 3.5 as in your code:
DEFAULT_CLASSIFY_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llm-extract")


# -----------------------------
# Small Utilities
# -----------------------------
def clean_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def combine_article_texts(article_folder: Path) -> str:
    """
    Joins numbered *.txt files (prefix like '0001_...') into one paragraph.
    """
    txts = sorted(
        [p for p in article_folder.glob("*.txt") if re.match(r"\d+_", p.name)],
        key=lambda x: int(re.match(r"(\d+)_", x.name).group(1)),
    )
    if not txts:
        log.warning("No text files found in %s", article_folder)
        return ""
    parts = [read_text_file(p).strip() for p in txts]
    return " ".join(parts).strip()


def detect_type_and_source(
    text: str,
    type_keywords: List[str],
    source_keywords: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight rule-based detector scanning the first couple of lines.
    """
    lines = clean_lines(text)
    scan_for_type = " ".join(lines[:2])
    article_type = next((t for t in type_keywords if t in scan_for_type), None)

    source_lines = lines[1:3] if article_type else lines[:2]
    article_source = None
    for ln in source_lines:
        hit = next((s for s in source_keywords if s in ln), None)
        if hit:
            article_source = hit
            break

    return article_type, article_source


# -----------------------------
# Bedrock helpers
# -----------------------------
def _bedrock_client(region: str = DEFAULT_REGION):
    return boto3.client("bedrock-runtime", region_name=region)


def _invoke_streaming(
    messages: List[Dict[str, Any]],
    *,
    model_id: str = DEFAULT_STREAM_MODEL_ID,
    region: str = DEFAULT_REGION,
    temperature: float = 0.2,
    max_tokens: int = 50_000,
    stop_sequences: Optional[List[str]] = None,
) -> str:
    """
    Calls Bedrock (Claude) with response streaming and returns concatenated text deltas.
    """
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop_sequences": stop_sequences or [],
        "anthropic_version": "bedrock-2023-05-31",
    }

    brt = _bedrock_client(region)
    resp = brt.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
        trace="ENABLED",
    )

    out = []
    for event in resp["body"]:
        if "chunk" in event:
            raw = event["chunk"]["bytes"].decode("utf-8")
            try:
                piece = json.loads(raw)
                delta = piece.get("delta", {})
                if txt := delta.get("text"):
                    out.append(txt)
            except Exception:
                # ignore partial JSON during stream
                pass
    return "".join(out)


def _extract_json_from_text(full_text: str) -> Dict[str, Any]:
    """
    Robustly pulls a single JSON object from a text (handles ```json fences).
    Returns dict; on failure returns {"error": ..., "raw": ...}.
    """
    txt = full_text.strip()
    # strip code fences if present
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL).strip()

    # prefer fenced JSON
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, re.DOTALL)
    if not m:
        m = re.search(r"```[\s\n]*(\{.*?\})[\s\n]*```", txt, re.DOTALL)
    if not m:
        # fallback: first {...} up to last closing brace
        m = re.search(r"(\{.*\})", txt, re.DOTALL)

    clean = (m.group(1) if m else txt).strip()
    first = clean.find("{")
    if first > 0:
        clean = clean[first:]

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output from model", "raw": clean}


# -----------------------------
# Generic Extractor
# -----------------------------
def extract_attributes_streaming(
    ocr_text: str,
    schema_path: str,
    *,
    model_id: str = DEFAULT_STREAM_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> Dict[str, Any]:
    """
    Generic streaming extractor: loads schema text and asks Claude to output strict JSON.
    """
    schema_description = read_text_file(Path(schema_path))

    prompt = f"""
أنت محلّل وثائق قانونية. استخرج JSON مطابقاً تماماً للمخطط التالي.
يجب تطبيق القواعد على كل الحقول (بما فيها محتوى المواد):

- حوّل كل الأرقام العربية إلى إنكليزية.
- صغ كل التواريخ بصيغة YYYY-MM-DD.
- لا تُدخل أي شرح خارج JSON.

المخطط:
{schema_description}

### النص:
{ocr_text}

### أعد فقط JSON واحد:
""".strip()

    full = _invoke_streaming(
        messages=[{"role": "user", "content": prompt}],
        model_id=model_id,
        region=region,
        temperature=0.2,
    )
    return _extract_json_from_text(full)


# -----------------------------
# Decrees (مراسيم) wrapper
# -----------------------------
def extract_decree_attributes_streaming(
    ocr_text: str,
    schema_path: str,
    *,
    model_id: str = DEFAULT_STREAM_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> Dict[str, Any]:
    """
    Wrapper; same as generic but keeps a separate entrypoint for clarity.
    """
    schema_description = read_text_file(Path(schema_path))
    prompt = f"""
{schema_description}

### Example Input:
{ocr_text}

### Output Format (in JSON):
""".strip()

    full = _invoke_streaming(
        messages=[{"role": "user", "content": prompt}],
        model_id=model_id,
        region=region,
        temperature=0.3,
    )
    return _extract_json_from_text(full)


# -----------------------------
# Karārāt (قرارات) extractor
# -----------------------------
def extract_attributes_streaming_kararat(
    subtype: str,
    ocr_text: str,
    *,
    model_id: str = DEFAULT_STREAM_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> Dict[str, Any]:
    """
    Currently supports subtype="قرار". Add schemas for other subtypes later.
    """
    subtype = (subtype or "").strip()
    if subtype != "قرار":
        return {"error": "Unsupported subtype", "subtype": subtype}

    schema_path = Path("karar_schema.txt")
    schema_description = read_text_file(schema_path)

    prompt = f"""
You are a legal document parser. You will receive the OCR text of a Lebanese قرار (karar / administrative decision). Extract the following fields:

{schema_description}

Do not hallucinate content. If a feature is missing from the article, return `null` for that feature (or an empty object/list if applicable).

### Example Input:
{ocr_text}

### Output Format (in JSON):
""".strip()

    full = _invoke_streaming(
        messages=[{"role": "user", "content": prompt}],
        model_id=model_id,
        region=region,
        temperature=0.3,
    )
    return _extract_json_from_text(full)


# -----------------------------
# Subtype detection (short, non-stream)
# -----------------------------
def detect_subtype_with_claude(
    text: str,
    *,
    region: str = DEFAULT_REGION,
    model_id: str = DEFAULT_CLASSIFY_MODEL_ID,
) -> str:
    """
    Classifies decision text into one of:
    قرار | بيان | إعلام | علم وخبر | بلاغ | قرار وسيط | قرار بلدي | unknown
    """
    client = _bedrock_client(region)
    system_prompt = """
You are a legal document analyzer. Given the full OCR text of a Lebanese decision article (قرارات تعاميم علم وخبر), classify it into one of:
- قرار
- بيان
- إعلام
- علم وخبر
- بلاغ
- قرار وسيط
- قرار بلدي

Respond with only the subtype (one word) or "unknown".
""".strip()

    payload = {
        "system": system_prompt,
        "messages": [{"role": "user", "content": text.strip()}],
        "max_tokens": 50,
        "anthropic_version": "bedrock-2023-05-31",
    }

    try:
        resp = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(resp["body"].read())
        return body.get("content", [{}])[0].get("text", "").strip() or "unknown"
    except Exception as e:
        log.warning("Subtype classification error: %s", e)
        return "unknown"
