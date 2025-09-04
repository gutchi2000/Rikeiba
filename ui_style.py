# ui_style.py  —— clean & minimal
from __future__ import annotations
import streamlit as st
from typing import Optional

CSS = """
:root{
  --brand:#4f46e5; --brand2:#14b8a6; --text:#e6e9f2;
  --bg:#0b1020; --card:#121a2d; --muted:#94a3b8;
}
section.main > div{padding-top:10px;}
.topbar{position:sticky;top:0;z-index:1000;background:linear-gradient(90deg,#4f46e5,#7c3aed);padding:14px 18px;border-radius:14px;margin-bottom:10px;box-shadow:0 12px 30px rgba(0,0,0,.25)}
.topbar h1{margin:0;font-size:18px;color:#fff;letter-spacing:.3px}
.topbar .sub{opacity:.9;font-size:12px;color:#eaf2ff}
.card{background:var(--card);border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:16px 18px;margin:8px 0;box-shadow:0 10px 25px rgba(0,0,0,.25)}
.card h3{margin:.2rem 0 .4rem;font-size:16px}
.stButton>button{background:linear-gradient(180deg,#4f46e5,#4338ca);color:#fff;border:none;border-radius:12px;padding:10px 14px;font-weight:600;box-shadow:0 6px 18px rgba(79,70,229,.35)}
.pill{display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:999px;font-weight:700;font-size:12px;border:1px solid rgba(255,255,255,.1)}
.pill.y{background:rgba(79,70,229,.18);color:#dfe3ff;border-color:rgba(79,70,229,.4)}
.pill.o{background:rgba(20,184,166,.18);color:#d7fff6;border-color:rgba(20,184,166,.45)}
.pill.s{background:rgba(59,130,246,.18);color:#e0f0ff;border-color:rgba(59,130,246,.45)}
.pill.d{background:rgba(148,163,184,.18);color:#e6e9f2;border-color:rgba(148,163,184,.35)}
.pill.x{background:rgba(239,68,68,.18);color:#ffe2e2;border-color:rgba(239,68,68,.4)}
.scorebar{position:relative;height:10px;background:#0f1830;border-radius:999px;border:1px solid rgba(255,255,255,.08);overflow:hidden}
.scorebar>span{position:absolute;inset:0;display:block;background:linear-gradient(90deg,var(--brand),var(--brand2));width:0%}
.tags{display:flex;gap:8px;flex-wrap:wrap}
.subtle{color:var(--muted);font-size:12px}
"""

def inject_css() -> None:
    if st.session_state.get("_ui_css_injected"):
        return
    st.session_state["_ui_css_injected"] = True
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

def topbar(title: str, subtitle: Optional[str] = None) -> None:
    inject_css()
    html = f"<div class='topbar'><h1>{title}</h1>"
    if subtitle:
        html += f"<div class='sub'>{subtitle}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def pill(mark: str) -> str:
    cls = {'◎':'y','○':'o','▲':'s','△':'d','×':'x'}.get(mark,'d')
    return f'<span class="pill {cls}"><i>{mark}</i></span>'

def score_bar(value: float, max_value: float = 100.0) -> str:
    pct = 0.0 if max_value <= 0 else max(0.0, min(100.0, (value/max_value)*100))
    return f'<div class="scorebar"><span style="width:{pct:.1f}%"></span></div>'

def card(title: str, right: Optional[str] = None):
    inject_css()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if right:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='text-align:right'>{right}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    return st.container()
