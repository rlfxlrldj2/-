import os, time, random, hashlib, re
from urllib.parse import urlparse
from datetime import datetime, date, time as dtime, timedelta

import pytz
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
import trafilatura
from tenacity import retry, wait_exponential, stop_after_attempt
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

# ===== 설정 =====
BASE_KEYWORDS = ["신축", "증설", "증축", "설비 증설", "시설 투자", "투자 협약", "투자 유치"]
KW_EXPANSIONS = {
    "신축":       ["신설", "신규 건립", "공장 신축", "신축공사"],
    "증설":       ["라인 증설", "규모 확대", "확충", "생산능력 확대", "증산"],
    "증축":       ["신·증축", "신증축", "증축공사"],
    "설비 증설":  ["설비증설", "설비 확충", "장비 증설", "라인 증설"],
    "시설 투자":  ["시설투자", "설비 투자", "설비투자", "대규모 투자", "공장 투자", "CAPEX"],
    "투자 협약":  ["투자협약", "투자 MOU", "양해각서", "MOU 체결", "유치 협약"],
    "투자 유치":  ["투자유치", "유치 협약", "유치 MOU", "대규모 유치"],
}
PER_BASE_LIMIT = 2          # 확장어를 각 기본 키워드당 몇 개까지 쓸지
ONLY_NAVER_DOMAIN = True    # 네이버 뉴스 도메인만 수집
MIN_CHARS = 200             # 본문 최소 길이
PER_QUERY_LIMIT = 120       # 쿼리당 최대 문서 수
TIMEZONE = "Asia/Seoul"

API_URL  = "https://openapi.naver.com/v1/search/news.json"
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise SystemExit("Missing NAVER_CLIENT_ID / NAVER_CLIENT_SECRET in env.")

API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
}
WEB_HEADERS = {k: v for k, v in API_HEADERS.items() if not k.startswith("X-Naver-")}

def compute_window_kst():
    kst = pytz.timezone(TIMEZONE)
    today_kst = datetime.now(kst).date()
    end_kst = kst.localize(datetime.combine(today_kst, dtime(7, 0, 0)))
    start_kst = end_kst - timedelta(days=1)
    return start_kst, end_kst

def _is_naver_host(url: str) -> bool:
    try:
        host = urlparse(url).netloc
        return any(h in host for h in [
            "news.naver.com", "n.news.naver.com", "sports.naver.com", "entertain.naver.com"
        ])
    except Exception:
        return False

def build_queries(base_keywords, expansions, per_base_limit=1):
    q = []
    for base in base_keywords:
        q.append(base)
        for i, alt in enumerate(expansions.get(base, [])):
            if i >= per_base_limit: break
            q.append(alt)
    seen, deduped = set(), []
    for s in q:
        if s not in seen:
            deduped.append(s); seen.add(s)
    return deduped

QUERIES = build_queries(BASE_KEYWORDS, KW_EXPANSIONS, per_base_limit=PER_BASE_LIMIT)

@sleep_and_retry
@limits(calls=20, period=60)
@retry(wait=wait_exponential(min=1, max=16), stop=stop_after_attempt(3))
def _search_once(query: str, start: int, display: int, sort: str):
    params = {"query": query, "start": start, "display": display, "sort": sort}
    r = requests.get(API_URL, headers=API_HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def collect_news_items_for_query(query: str, sort: str = "date", limit: int = 200):
    items, start = [], 1
    while len(items) < limit and start <= 1000:
        display = min(100, limit - len(items))
        data = _search_once(query, start=start, display=display, sort=sort)
        batch = data.get("items", [])
        if not batch: break
        items.extend(batch)
        start += len(batch)
        time.sleep(random.uniform(0.2, 0.6))
    return items[:limit]

def collect_all_items(queries, per_query_limit=200):
    all_items = []
    for q in queries:
        try:
            all_items.extend(collect_news_items_for_query(q, sort="date", limit=per_query_limit))
        except Exception as e:
            print("[API 오류]", q, e)
    return all_items

def _clean_text(s: str) -> str:
    if not s: return ""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())

def extract_article_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    candidates = ["#dic_area", "#newsct_article", "article"]
    for css in candidates:
        node = soup.select_one(css)
        if not node: 
            continue
        for junk in node.select("script, style, noscript, figure, table, aside, .byline, .copyright"):
            junk.decompose()
        texts = [t.get_text(" ", strip=True) for t in node.find_all(["p","div","span"]) if t.get_text(strip=True)]
        text = _clean_text("\n".join(texts))
        if len(text) > 100:
            return text
    return _clean_text(soup.get_text(" ", strip=True))

@sleep_and_retry
@limits(calls=30, period=60)
@retry(wait=wait_exponential(min=1, max=16), stop=stop_after_attempt(3))
def fetch_html(url: str):
    r = requests.get(url, headers=WEB_HEADERS, timeout=20, allow_redirects=True)
    r.raise_for_status()
    return r.text, r.url

def extract_fulltext(url: str):
    html, final_url = fetch_html(url)
    text = trafilatura.extract(html, url=final_url, include_comments=False, include_tables=False, favor_recall=True)
    if text and len(text.split()) > 30:
        return _clean_text(text), final_url
    return extract_article_text_from_html(html), final_url

def _make_kw_regex(w: str) -> str:
    parts = []
    for ch in w:
        if ch.isspace():
            parts.append(r"\s*")
        elif ch in ("·", "ㆍ"):
            parts.append(r"[·ㆍ]?")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)

ALL_KW_VARIANTS = set(BASE_KEYWORDS)
for base, alts in KW_EXPANSIONS.items():
    for a in alts:
        ALL_KW_VARIANTS.add(a)

KW_VARIANTS_SORTED = sorted(list(ALL_KW_VARIANTS), key=len, reverse=True)
KW_PATTERNS = [re.compile(_make_kw_regex(w)) for w in KW_VARIANTS_SORTED]
CANONICAL_PATTERNS = {tag: re.compile(_make_kw_regex(tag)) for tag in BASE_KEYWORDS}

def find_matched_terms(text: str) -> str:
    mapped = [tag for tag, pat in CANONICAL_PATTERNS.items() if pat.search(text)]
    if mapped:
        return ", ".join(sorted(set(mapped)))
    hits = [p.pattern for p in KW_PATTERNS if p.search(text)]
    return ", ".join(sorted(set(hits)))

def main_run():
    kst = pytz.timezone(TIMEZONE)
    START_KST, END_KST = compute_window_kst()
    items = collect_all_items(QUERIES, PER_QUERY_LIMIT)
    rows, seen = [], set()

    for it in tqdm(items, desc="본문 추출/정리"):
        title = BeautifulSoup(it.get("title", ""), "lxml").get_text(" ", strip=True)
        desc  = BeautifulSoup(it.get("description", ""), "lxml").get_text(" ", strip=True)
        pub   = it.get("pubDate", "")
        try:
            pub_dt = dtparser.parse(pub) if pub else None
            if pub_dt and pub_dt.tzinfo:
                pub_kst = pub_dt.astimezone(kst)
            else:
                pub_kst = kst.localize(pub_dt) if pub_dt else None
        except Exception:
            pub_dt = pub_kst = None

        if pub_kst and not (START_KST <= pub_kst < END_KST):
            continue

        link   = it.get("link") or ""
        origin = it.get("originallink") or ""
        chosen = link if (_is_naver_host(link)) else (link or origin or "")
        if ONLY_NAVER_DOMAIN and not _is_naver_host(chosen):
            if _is_naver_host(link):
                chosen = link
            else:
                continue
        if not chosen:
            continue

        try:
            text, final_url = extract_fulltext(chosen)
        except Exception:
            if origin and origin != chosen:
                try:
                    text, final_url = extract_fulltext(origin)
                except Exception:
                    text, final_url = "", chosen
            else:
                text, final_url = "", chosen

        if len(text) < MIN_CHARS:
            continue

        key = hashlib.md5(final_url.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        full_for_match = (title + "\n" + desc + "\n" + text)
        matched = find_matched_terms(full_for_match)

        rows.append({
            "pubDate_kst": pub_kst,
            "title": title,
            "naver_link": link,
            "originallink": origin,
            "used_url": final_url,
            "text_len": len(text),
            "matched_tags": matched,
            "snippet": text[:240].replace("\n", " ") + ("..." if len(text) > 240 else ""),
            "text": text,
        })
        time.sleep(random.uniform(0.1, 0.3))

    df = pd.DataFrame(rows).sort_values(by="pubDate_kst", ascending=False, na_position="last")
    stamp = datetime.now().strftime("%Y%m%d")
    out_csv = f"naver_news_{stamp}_07to07.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[완료] 수집 {len(df)}건 → 저장: {out_csv}")
    return out_csv, df

if __name__ == "__main__":
    main_run()
