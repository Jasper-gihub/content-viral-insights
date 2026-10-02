# -*- coding: utf-8 -*-
"""02_analyze.py —— 内容爆款规律分析（B 站「数据分析 / 求职」赛道 806 条视频）。

思路：
  1. 播放量高度长尾 → 用分位数而不是均值定义「爆款」（Top 10%）与「普通」（后 50%）
  2. 把标题拆成可判定的特征（长度 / 数字 / 疑问词 / 第一人称 / 时效词 / 情绪强调词）
     对比特征在两组中的出现率，用 lift（爆款组出现率 ÷ 普通组出现率）衡量区分度
  3. 时长、互动率（点赞率 / 收藏率）同样做两组对比
  4. 输出可直接执行的内容建议

产出：
  reports/analysis_results.csv    特征 lift 表
  reports/figures/fig1_播放分布.png
  reports/figures/fig2_标题特征lift.png
  reports/figures/fig3_时长与互动率.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "cleaned" / "videos.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY, BLUE, LIGHT, PALE = "#1B3A5C", "#2E6DA4", "#A9C6E0", "#DCE8F4"
ORANGE, RED, GREY, INK = "#E1812C", "#C0392B", "#94A3B3", "#24384B"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C9D6E2", "axes.linewidth": 0.9,
    "axes.labelcolor": INK, "axes.titlecolor": NAVY,
    "xtick.color": "#5A6B7B", "ytick.color": "#5A6B7B",
    "axes.grid": True, "grid.color": "#E8EEF4", "grid.linewidth": 0.9,
    "axes.axisbelow": True, "font.size": 11, "legend.frameon": False,
})


def clean_ax(ax, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x" if xgrid else "y")


FEATURES = {
    "标题含数字": lambda t: bool(re.search(r"\d", t)),
    "标题含疑问词（怎么/如何/为什么）": lambda t: bool(re.search(r"怎么|如何|为什么|凭什么", t)),
    "标题含第一人称（我/亲测/血泪）": lambda t: bool(re.search(r"我|亲测|血泪|实录", t)),
    "标题含时效词（2025/2026/最新）": lambda t: bool(re.search(r"20(2[4-9])|最新|新版", t)),
    "标题含强情绪词（必看/干货/避坑/真相）": lambda t: bool(re.search(r"必看|干货|避坑|真相|保姆|零基础|建议收藏|劝退", t)),
    "标题含感叹/问号": lambda t: bool(re.search(r"[！？!?]", t)),
    "标题长度 ≥ 20 字": lambda t: len(str(t)) >= 20,
}


def main() -> None:
    df_all = pd.read_csv(CLEAN)
    df_all = df_all[df_all["play"].notna() & (df_all["play"] > 0)].copy()
    df_all["title"] = df_all["title"].fillna("").astype(str)

    # ---- 结构发现：赛道头部由「合集/全套课程」主导（机构出品）
    top_all = df_all.nlargest(int(len(df_all) * 0.1), "play")
    coll_share = (top_all["duration_sec"] > 14400).mean()
    print(f"全样本 {len(df_all)} 条 ｜ Top10% 中 {coll_share:.1%} 是 4 小时以上的合集/全套课程")
    print(f"  其中 >30 分钟占 {(top_all['duration_sec'] > 1800).mean():.1%}")

    # ---- 主分析样本：单集内容型视频（≤30 分钟）—— 个人创作者可对标的口径
    df = df_all[df_all["duration_sec"] <= 1800].copy()
    print(f"主分析样本（单集内容型 ≤30 分钟）：{len(df)} 条")

    p50, p90 = df["play"].quantile([0.5, 0.9])
    hot = df[df["play"] >= p90]
    cold = df[df["play"] <= p50]
    print(f"播放中位数 {p50:,.0f} ｜ P90（爆款线）{p90:,.0f}")
    print(f"爆款组 {len(hot)} 条（Top 10%）｜ 普通组 {len(cold)} 条（后 50%）")
    print(f"爆款组播放中位数 {hot['play'].median():,.0f}，是普通组的 {hot['play'].median() / max(cold['play'].median(),1):.0f} 倍")

    # ---------------------------------------------------------------- 特征 lift
    rows = []
    for name, fn in FEATURES.items():
        h = np.mean([bool(fn(t)) for t in hot["title"]])
        c = np.mean([bool(fn(t)) for t in cold["title"]])
        rows.append({"特征": name, "爆款组出现率": h, "普通组出现率": c,
                     "lift": h / c if c else float("inf")})
    # 时长与互动率作为数值特征单独比较
    rows.append({"特征": "时长 ≤ 10 分钟", "爆款组出现率": (hot["duration_sec"] <= 600).mean(),
                 "普通组出现率": (cold["duration_sec"] <= 600).mean(),
                 "lift": (hot["duration_sec"] <= 600).mean() / max((cold["duration_sec"] <= 600).mean(), 1e-9)})
    feat = pd.DataFrame(rows).sort_values("lift", ascending=False)
    feat["lift"] = feat["lift"].round(2)
    feat["爆款组出现率"] = (feat["爆款组出现率"] * 100).round(1)
    feat["普通组出现率"] = (feat["普通组出现率"] * 100).round(1)

    # 关键指标对比
    compare = pd.DataFrame({
        "爆款组": {"标题平均字数": hot["title"].str.len().mean(), "时长中位数(分钟)": hot["duration_sec"].median() / 60,
                   "点赞率": hot["like_rate"].median(), "收藏率": hot["fav_rate"].median()},
        "普通组": {"标题平均字数": cold["title"].str.len().mean(), "时长中位数(分钟)": cold["duration_sec"].median() / 60,
                   "点赞率": cold["like_rate"].median(), "收藏率": cold["fav_rate"].median()},
    }).round(3)
    print("\n== 关键指标对比 ==")
    print(compare.to_string())
    print("\n== 特征 lift（>1 = 与爆款正相关）==")
    print(feat.to_string(index=False))

    feat.to_csv(ROOT / "reports" / "analysis_results.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(ROOT / "reports" / "group_compare.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- 图1 播放分布
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    bins = np.logspace(np.log10(df["play"].clip(lower=1).min()), np.log10(df["play"].max()), 40)
    ax.hist(df["play"], bins=bins, color=BLUE, alpha=0.85, zorder=3)
    ax.set_xscale("log")
    ax.axvline(p50, color=ORANGE, ls=(0, (4, 3)), lw=1.6)
    ax.axvline(p90, color=RED, ls=(0, (4, 3)), lw=1.6)
    ax.text(p50, ax.get_ylim()[1] * 0.92, f" 中位数 {p50:,.0f}", color=ORANGE, fontsize=10.5,
            fontweight="bold")
    ax.text(p90, ax.get_ylim()[1] * 0.78, f" 爆款线 P90 = {p90:,.0f}", color=RED, fontsize=10.5,
            fontweight="bold")
    ax.set_xlabel("播放量（对数刻度）", fontsize=10.5)
    ax.set_ylabel("视频数", fontsize=10.5)
    clean_ax(ax)
    ax.text(0.02, 0.80, f"口径：单集内容型视频（≤30 分钟），共 {len(df)} 条\n"
                        f"全样本 Top10% 播放中 {coll_share:.0%} 是 4 小时以上的\n"
                        f"合集/全套课程（机构出品）——个人创作者需另设口径",
            transform=ax.transAxes, ha="left", fontsize=9.2, color=GREY, linespacing=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_播放分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 图2 特征 lift
    f2 = feat.sort_values("lift")
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    y = np.arange(len(f2))
    colors = [RED if v >= 1.5 else ORANGE if v >= 1.1 else GREY for v in f2["lift"]]
    ax.barh(y, f2["lift"], height=0.6, color=colors, zorder=3)
    ax.axvline(1.0, color=GREY, lw=1.2)
    for yi, (v, h, c) in zip(y, zip(f2["lift"], f2["爆款组出现率"], f2["普通组出现率"])):
        ax.text(v + 0.03, yi, f"lift {v:.2f}（爆款 {h:.0f}% vs 普通 {c:.0f}%）",
                va="center", fontsize=9.4, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(f2["特征"], fontsize=10)
    ax.set_xlim(0, max(f2["lift"]) * 1.55)
    ax.set_xlabel("lift = 爆款组出现率 ÷ 普通组出现率（1.0 = 无区分度）", fontsize=10.5)
    clean_ax(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_标题特征lift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 图3 时长与互动率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    buckets = [(0, 180, "≤3 分钟"), (180, 600, "3–10 分钟"), (600, 1800, "10–30 分钟")]
    names, med = [], []
    for lo, hi, label in buckets:
        g = df[(df["duration_sec"] > lo) & (df["duration_sec"] <= hi)]
        if len(g) >= 10:
            names.append(f"{label}\n(n={len(g)})")
            med.append(g["play"].median())
    ax1.bar(names, med, color=[LIGHT, BLUE, ORANGE][:len(names)], width=0.6, zorder=3)
    for i, v in enumerate(med):
        ax1.text(i, v * 1.06, f"{v:,.0f}", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax1.set_ylim(0, max(med) * 1.22)
    ax1.set_ylabel("播放量中位数", fontsize=10.5)
    ax1.set_title("时长与播放中位数", fontsize=11.5, pad=8)
    clean_ax(ax1)

    labels = ["点赞率", "收藏率"]
    h_vals = [hot["like_rate"].median() * 100, hot["fav_rate"].median() * 100]
    c_vals = [cold["like_rate"].median() * 100, cold["fav_rate"].median() * 100]
    x = np.arange(len(labels))
    ax2.bar(x - 0.18, h_vals, width=0.34, color=RED, label="爆款组", zorder=3)
    ax2.bar(x + 0.18, c_vals, width=0.34, color=LIGHT, label="普通组", zorder=3)
    for xi, v in zip(x - 0.18, h_vals):
        ax2.text(xi, v + 0.06, f"{v:.1f}%", ha="center", fontsize=10, color=RED, fontweight="bold")
    for xi, v in zip(x + 0.18, c_vals):
        ax2.text(xi, v + 0.06, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("占播放量的比例（中位数，%）", fontsize=10.5)
    ax2.set_title("互动率：爆款 vs 普通", fontsize=11.5, pad=8)
    ax2.legend(fontsize=10)
    clean_ax(ax2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_时长与互动率.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n已写出：{ROOT / 'reports' / 'analysis_results.csv'}")
    print(f"图表：{FIG}")


if __name__ == "__main__":
    main()
