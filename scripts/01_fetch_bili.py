# -*- coding: utf-8 -*-
"""01_fetch_bili.py —— 抓取 B 站「数据分析 / 求职」类视频的公开数据。

数据源：B 站公开搜索接口（无需登录，国内直连）
    https://api.bilibili.com/x/web-interface/search/type
用途：为「内容爆款规律分析」提供样本——用播放/点赞/收藏等公开指标，
      研究高播放视频在标题、时长、发布时间上的共同特征。

用法：python scripts/01_fetch_bili.py
产出：data/raw/bili_search.json（原始响应）
      data/cleaned/videos.csv（视频明细：标题/UP/播放/点赞/收藏/弹幕/时长/发布时间）
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://www.bilibili.com/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

KEYWORDS = ["数据分析 求职", "数据分析 转行", "SQL 学习", "Python 数据分析", "数据分析师 入门"]
MAX_PAGES = 10          # 每关键词最多 10 页（每页 20 条）
PAGE_SLEEP = 1.6

EM_RE = re.compile(r"</?em[^>]*>")


def make_session() -> requests.Session:
    """建立带 cookie 的会话：先访问首页拿 buvid3，再调搜索接口（显著降低被反爬概率）。"""
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.bilibili.com/", timeout=15)
    except Exception as e:                                       # noqa: BLE001
        print(f"  [warn] 首页 cookie 获取失败：{type(e).__name__} {e}")
    return s


def fetch_keyword(session: requests.Session, kw: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    raw: dict[str, dict] = {}
    for page in range(1, MAX_PAGES + 1):
        url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {"search_type": "video", "keyword": kw, "page": page, "page_size": 20}
        d = None
        for attempt in range(1, 4):
            try:
                r = session.get(url, params=params, timeout=20)
                d = r.json()
                if d.get("code") == 0:
                    break
                print(f"  [retry {attempt}] {kw} 第 {page} 页 code={d.get('code')}")
            except Exception as e:                               # noqa: BLE001
                print(f"  [retry {attempt}] {kw} 第 {page} 页 {type(e).__name__}")
            time.sleep(3 * attempt)
        if not d or d.get("code") != 0:
            print(f"  [warn] {kw} 第 {page} 页最终失败，跳过")
            break
        raw[f"page_{page}"] = d
        items = (d.get("data") or {}).get("result") or []
        if not items:
            break
        for it in items:
            rows.append({
                "keyword": kw,
                "bvid": it.get("bvid"),
                "title": EM_RE.sub("", str(it.get("title", ""))),
                "author": it.get("author"),
                "play": it.get("play"),
                "like": it.get("like"),
                "favorites": it.get("favorites"),
                "danmaku": it.get("danmaku"),
                "review": it.get("review"),
                "duration": it.get("duration"),
                "pubdate": it.get("pubdate"),
                "arcurl": it.get("arcurl"),
            })
        print(f"  {kw} 第 {page} 页：{len(items)} 条")
        time.sleep(PAGE_SLEEP)
    return rows, raw


def parse_duration(s: str) -> int:
    """'3:55' / '1:02:33' -> 秒。"""
    try:
        parts = [int(p) for p in str(s).split(":")]
    except ValueError:
        return 0
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_raw: dict[str, dict] = {}
    session = make_session()
    for kw in KEYWORDS:
        print(f"\n== {kw} ==")
        rows, raw = fetch_keyword(session, kw)
        all_rows.extend(rows)
        all_raw[kw] = raw
        time.sleep(4.0)                       # 关键词之间多等一会，降低风控概率

    (RAW_DIR / "bili_search.json").write_text(
        json.dumps(all_raw, ensure_ascii=False, indent=1), encoding="utf-8")

    df = pd.DataFrame(all_rows)
    before = len(df)
    df = df.drop_duplicates(subset=["bvid"]).reset_index(drop=True)
    df["duration_sec"] = df["duration"].map(parse_duration)
    df["pubdate"] = pd.to_datetime(df["pubdate"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    df["pub_year_month"] = df["pubdate"].dt.strftime("%Y-%m")
    df["like_rate"] = (df["like"] / df["play"].clip(lower=1)).round(4)
    df["fav_rate"] = (df["favorites"] / df["play"].clip(lower=1)).round(4)

    out = CLEAN_DIR / "videos.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n合计 {before} 条，去重后 {len(df)} 条")
    print(df.groupby("keyword").agg(条数=("bvid", "size"), 播放中位数=("play", "median")).to_string())
    print(f"\n已写出：{out}")


if __name__ == "__main__":
    main()
