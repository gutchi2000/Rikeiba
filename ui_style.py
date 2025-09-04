# ui_style.py
}


/* ピル（バッジ） */
.pill{display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:999px;font-weight:700;font-size:12px;border:1px solid rgba(255,255,255,.1);}
.pill i{font-style:normal;opacity:.9}
.pill.y{background:rgba(79,70,229,.18);color:#dfe3ff;border-color:rgba(79,70,229,.4)} /* ◎ */
.pill.o{background:rgba(20,184,166,.18);color:#d7fff6;border-color:rgba(20,184,166,.45)} /* ○ */
.pill.s{background:rgba(59,130,246,.18);color:#e0f0ff;border-color:rgba(59,130,246,.45)} /* ▲ */
.pill.d{background:rgba(148,163,184,.18);color:#e6e9f2;border-color:rgba(148,163,184,.35)} /* △ */
.pill.x{background:rgba(239,68,68,.18); color:#ffe2e2;border-color:rgba(239,68,68,.4)} /* × */


/* スコアバー */
.scorebar{position:relative;height:10px;background:#0f1830;border-radius:999px;border:1px solid rgba(255,255,255,.08);overflow:hidden}
.scorebar>span{position:absolute;inset:0 0 0 0;display:block;background:linear-gradient(90deg,var(--brand),var(--brand-2));width:0%}
.scoreval{font-variant-numeric:tabular-nums;font-weight:700;}


/* タグ群の横スクロール */
.tags{display:flex;gap:8px;flex-wrap:wrap}


/* 小さめのセカンダリテキスト */
.subtle{color:var(--muted);font-size:12px}
'''




def inject_css():
"""一度だけCSSを注入する。"""
if 'ui_css_injected' in st.session_state:
return
st.session_state['ui_css_injected'] = True
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)




# ====== UI ヘルパ ======


def topbar(title: str, subtitle: Optional[str] = None):
inject_css()
st.markdown(
f"""
<div class="topbar">
<h1>{title}</h1>
{f'<div class="sub">{subtitle}</div>' if subtitle else ''}
</div>
""", unsafe_allow_html=True
)




def pill(mark: str) -> str:
"""印からスタイル済みピルHTMLを返す。◎○▲△×に対応。"""
cls = {'◎':'y','○':'o','▲':'s','△':'d','×':'x'}.get(mark,'d')
return f'<span class="pill {cls}"><i>{mark}</i></span>'




def score_bar(value: float, max_value: float = 100.0) -> str:
"""スコアの横棒。value/max_value で幅を決める。"""
pct = max(0.0, min(100.0, (value/max_value)*100))
return f'<div class="scorebar"><span style="width:{pct:.1f}%"></span></div>'




def card(title: str, right: Optional[str] = None):
"""with で使うカードコンテナ（終了は st.container の挙動）。"""
inject_css()
st.markdown('<div class="card">', unsafe_allow_html=True)
cols = st.columns([1,1]) if right else None
if right:
with cols[0]:
st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
with cols[1]:
st.markdown(f"<div style='text-align:right;'>{right}</div>", unsafe_allow_html=True)
else:
st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
return st.container()
