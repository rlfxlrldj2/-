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

# ===================== 사용자 요건 반영 설정 =====================

# 기준 키워드 (요청)
BASE_KEYWORDS = ["신축", "증설", "증축", "설비 증설", "시설 투자", "투자 협약", "투자 유치"]

# 유사어/변형 (요청)
KW_EXPANSIONS = {
    "신축":       ["공장 신설", "공장 신축", "공장 신규", "신축공사"],
    "증설":       ["공장 증설", "규모 확대", "확충", "생산능력 확대", "증설공사"],
    "증축":       ["공장 증축", "증축 공사", "증축공사"],
    "설비 증설":  ["설비증설", "장비 증설", "라인 증설", "증설 공사"],
    "시설 투자":  ["시설투자", "시설 투자", "설비투자", "공장 투자"],
    "투자 협약":  ["투자협약", "투자 MOU", "공장 투자"],
    "투자 유치":  ["투자유치", "유치 협약", "투자 유치"],
}

# 검색 규모(필요 시 조절)
PER_BASE_LIMIT   = 2     # 각 기본 키워드당 확장어 몇 개까지 검색에 사용할지
PER_QUERY_LIMIT  = 40   # 쿼리당 최대 기사 수
ONLY_NAVER_DOMAIN = True # 네이버 뉴스 도메인만 사용
MIN_CHARS        = 200   # 본문 최소 길이(짧은 페이지 제외)
TIMEZONE         = "Asia/Seoul"

# ===================== 네이버 검색 Open API =====================

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
    """전날 07:00 ~ 오늘 07:00 (KST)"""
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
    # 순서 유지 중복 제거
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

# ===================== 본문 추출 =====================

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
    return r.text, r.url  # 최종 URL

def extract_fulltext(url: str):
    html, final_url = fetch_html(url)
    text = trafilatura.extract(html, url=final_url, include_comments=False, include_tables=False, favor_recall=True)
    if text and len(text.split()) > 30:
        return _clean_text(text), final_url
    return extract_article_text_from_html(html), final_url

# ===================== 키워드 매칭(요청 키워드/유사어) =====================

def _make_kw_regex(w: str) -> str:
    """공백/중간점 허용 정규식(치환문자 \s 문제 방지)"""
    parts = []
    for ch in w:
        if ch.isspace():
            parts.append(r"\s*")
        elif ch in ("·", "ㆍ"):
            parts.append(r"[·ㆍ]?")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)

# per-tag 패턴
TAG_PATTERNS = {}
for tag in BASE_KEYWORDS:
    pats = [re.compile(_make_kw_regex(tag))]
    for alt in KW_EXPANSIONS.get(tag, []):
        pats.append(re.compile(_make_kw_regex(alt)))
    TAG_PATTERNS[tag] = pats

def find_tags_for_article(title: str, snippet: str, text: str):
    """한 기사에서 매칭되는 모든 태그(키워드)를 반환"""
    full = f"{title}\n{snippet}\n{text}"
    hit_tags = []
    for tag in BASE_KEYWORDS:  # 요청한 순서를 우선순위로 유지
        pats = TAG_PATTERNS[tag]
        if any(p.search(title) for p in pats) or any(p.search(full) for p in pats):
            hit_tags.append(tag)
    return hit_tags

# ===================== 필드 추출(요청 양식) =====================

PHONE_RE = re.compile(r"(0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}")

AMOUNT_PATTERNS = [
    re.compile(r"총?\s*(?:투자(?:금액|비)?|사업비|총사업비)\s*[:은]?\s*([\d,\.]+\s*(?:조|억)?\s*원)"),
    re.compile(r"(\d+)\s*조\s*(\d{1,3}(?:,\d{3})*)\s*억\s*원"),
    re.compile(r"([\d,\.]+)\s*조\s*원"),
    re.compile(r"([\d,\.]+)\s*억\s*원"),
]

SCALE_PATTERNS = [
    re.compile(r"규모[^\n]{0,20}?([0-9,\.]+\s*(?:㎡|m²|평|MW|GW|라인|대|톤|t|기|명))"),
    re.compile(r"연면적\s*[:은]?\s*([0-9,\.]+\s*(?:㎡|m²|평))"),
    re.compile(r"(?:부지|대지)\s*(?:면적)?\s*[:은]?\s*([0-9,\.]+\s*(?:㎡|m²|평))"),
    re.compile(r"(?:생산|CAPA|캐파)[^\n]{0,10}?([0-9,\.]+\s*(?:톤|t|K|k|만|억|라인))"),
]

LOC_RE = re.compile(
    r"(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|"
    r"경기도|강원특별자치도|강원도|충청북도|충북|충청남도|충남|전라북도|전북|전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주)"
    r"(?:\s*[가-힣0-9]{1,10}(?:시|군|구))?(?:\s*[가-힣0-9]{1,10}(?:읍|면|동|리))?"
)

CLIENT_PATTERNS = [
    re.compile(r"발주(?:처)?\s*[:는]\s*([^\n,.;()]+)"),
    re.compile(r"(주식회사\s*[가-힣A-Za-z0-9·\(\)\s]+|㈜\s*[가-힣A-Za-z0-9·\(\)\s]+|\(주\)\s*[가-힣A-Za-z0-9·\(\)\s]+)"),
    re.compile(r"[가-힣A-Za-z0-9·\(\)\s]{2,30}(?:공사|공단|개발공사|개발공단|도청|시청|군청|구청|경제자유구역청|테크노파크|산업단지공단)"),
]

def extract_first(patterns, text):
    for p in patterns:
        m = p.search(text)
        if m:
            g = m.groups() if hasattr(m, "groups") else None
            try:
                if g and len(g) > 1:
                    return " ".join([x for x in g if x])
                if g and len(g) == 1:
                    return g[0]
                return m.group(1) if m.lastindex else m.group(0)
            except Exception:
                return m.group(0)
    return ""

def extract_investment(text):
    for p in AMOUNT_PATTERNS:
        m = p.search(text)
        if not m:
            continue
        if len(m.groups()) >= 2 and m.group(2):
            return f"{m.group(1)}조 {m.group(2)}억 원"
        return m.group(1).strip() if m.groups() else m.group(0).strip()
    return ""

def extract_scale(text):
    return extract_first(SCALE_PATTERNS, text)

def extract_phone(text):
    m = PHONE_RE.search(text)
    return m.group(0) if m else ""

def extract_location(text):
    m = LOC_RE.search(text)
    return m.group(0) if m else ""

def extract_client(title, text):
    candidate = extract_first(CLIENT_PATTERNS, text)
    if candidate:
        return candidate.strip()
    candidate = extract_first(CLIENT_PATTERNS, title)
    return candidate.strip()

def summarize_100(text):
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    return s[:100]

# ===================== 메인 파이프라인 =====================

def main_run():
    kst = pytz.timezone(TIMEZONE)
    START_KST, END_KST = compute_window_kst()

    # 1) 기사 URL 수집
    items = collect_all_items(QUERIES, PER_QUERY_LIMIT)

    rows, seen_final_url = [], set()

    # 2) 본문 추출 + 시간 필터 + 태깅
    for it in tqdm(items, desc="본문 추출/정리"):
        title = BeautifulSoup(it.get("title", ""), "lxml").get_text(" ", strip=True)
        desc  = BeautifulSoup(it.get("description", ""), "lxml").get_text(" ", strip=True)
        pub   = it.get("pubDate", "")
        try:
            pub_dt = dtparser.parse(pub) if pub else None
            pub_kst = pub_dt.astimezone(kst) if pub_dt and pub_dt.tzinfo else (kst.localize(pub_dt) if pub_dt else None)
        except Exception:
            pub_dt = pub_kst = None

        # 시간창: 전날 07:00 ~ 오늘 07:00
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
            # 원문으로 재시도
            if origin and origin != chosen:
                try:
                    text, final_url = extract_fulltext(origin)
                except Exception:
                    text, final_url = "", chosen
            else:
                text, final_url = "", chosen

        if len(text) < MIN_CHARS:
            continue

        # 3) 키워드 태그 부여(여러 개 가능)
        tags = find_tags_for_article(title, desc, text)
        if not tags:
            continue

        rows.append({
            "pubDate_kst": pub_kst,
            "title": title,
            "snippet": desc,
            "used_url": final_url,
            "text": text,
            "matched_tags": tags,  # list
        })
        time.sleep(random.uniform(0.1, 0.3))

    if not rows:
        # 빈 결과라도 CSV/엑셀 경로는 리턴
        stamp = datetime.now().strftime("%Y%m%d")
        out_csv = f"naver_news_{stamp}_07to07.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        out_xlsx = f"naver_news_{stamp}_07to07.report.xlsx"
        with pd.ExcelWriter(out_xlsx) as w:
            pd.DataFrame().to_excel(w, index=False, sheet_name="전체")
        print("[완료] 수집 0건")
        return out_csv, out_xlsx

    df = pd.DataFrame(rows)

    # 4) 요청 포맷으로 정규화 컬럼 만들기
    def build_row(row):
        title = str(row.get("title","")).strip()
        text  = str(row.get("text","")).strip()
        snippet = str(row.get("snippet","")).strip()
        full = f"{title}\n{snippet}\n{text}"

        client = extract_client(title, text) if text else extract_client(title, snippet)
        project_name = title  # 보수적으로 제목 사용
        summary_100 = summarize_100(text or snippet)
        contact = extract_phone(full)
        scale = extract_scale(full)
        invest = extract_investment(full)
        location = extract_location(full)
        url = row.get("used_url") or ""

        return pd.Series({
            "발주처 이름": client,
            "프로젝트 이름": project_name,
            "내용 요약(100자 이내)": summary_100,
            "담당자 연락처": contact,
            "규모": scale,
            "투자금액": invest,
            "공사지역": location,
            "원본기사 URL 링크": url
        })

    norm = df.apply(build_row, axis=1)
    df_norm = pd.concat([df[["pubDate_kst","title","matched_tags","used_url"]], norm], axis=1)

    # 5) CSV 저장(원문 전체 표)
    stamp = datetime.now().strftime("%Y%m%d")
    out_csv = f"naver_news_{stamp}_07to07.csv"
    df_norm.to_csv(out_csv, index=False, encoding="utf-8-sig")

   # 6) 엑셀 저장 — 키워드별 시트 (각 시트 내 URL 중복 제거)
out_xlsx = f"naver_news_{stamp}_07to07.report.xlsx"
cols_order = ["발주처 이름","프로젝트 이름","내용 요약(100자 이내)","담당자 연락처","규모","투자금액","공사지역","원본기사 URL 링크"]

# ---- NEW: Excel 친화적 시각 열 만들기 (tz 제거) ----
df_norm_excel = df_norm.copy()
if "pubDate_kst" in df_norm_excel.columns:
    # 1) Series로 안전하게 datetime 변환 (tz-aware/naive 섞여 있어도 처리)
    dt = pd.to_datetime(df_norm_excel["pubDate_kst"], errors="coerce", utc=True)
    # 2) UTC -> KST로 변환 후 tz정보 제거(naive)
    dt_kst_naive = dt.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df_norm_excel["게재시각(KST)"] = dt_kst_naive
else:
    df_norm_excel["게재시각(KST)"] = ""

with pd.ExcelWriter(out_xlsx) as writer:
    # 전체 시트(원하는 열 + 게재시각)
    all_df = df_norm_excel[cols_order + ["게재시각(KST)","title","matched_tags"]].copy()
    all_df.to_excel(writer, index=False, sheet_name="전체")

    # 키워드별 시트
    for tag in BASE_KEYWORDS:
        mask = df_norm_excel["matched_tags"].apply(lambda lst: tag in lst if isinstance(lst, list) else False)
        sub = df_norm_excel[mask].copy()
        # 같은 키워드 시트 안에서 URL 중복 제거
        sub = sub.drop_duplicates(subset=["원본기사 URL 링크"])
        sub = sub[cols_order + ["게재시각(KST)"]]  # 게재시각 포함
        sub.to_excel(writer, index=False, sheet_name=tag[:31])

print(f"[완료] 기사 {len(df_norm)}건 → CSV: {out_csv}, XLSX: {out_xlsx}")
return out_csv, out_xlsx

if __name__ == "__main__":
    try:
        csv, xlsx = main_run()
    except Exception as e:
        print("[ERROR] Crawler failed:", repr(e))
        import traceback, pandas as pd
        traceback.print_exc()
        # 실패해도 비어있는 산출물을 만들어 워크플로는 성공 처리
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d")
        empty_csv = f"naver_news_{stamp}_07to07.csv"
        pd.DataFrame().to_csv(empty_csv, index=False, encoding="utf-8-sig")
        empty_xlsx = f"naver_news_{stamp}_07to07.report.xlsx"
        with pd.ExcelWriter(empty_xlsx) as w:
            pd.DataFrame().to_excel(w, index=False, sheet_name="전체")
        print("[INFO] Wrote empty outputs:", empty_csv, empty_xlsx)
        raise SystemExit(0)
