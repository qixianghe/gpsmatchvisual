"""
Football Match GPS Dashboard
=============================
Run with:  streamlit run app.py
"""

import hmac
import io
import os
import re
import zlib
from datetime import datetime

import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Match GPS Dashboard", layout="wide", page_icon="⚽")


# --------------------------------------------------------------------------
# Password gate
# --------------------------------------------------------------------------
def check_password():
    """Simple shared-password gate. Password comes from st.secrets['APP_PASSWORD']
    (set locally in .streamlit/secrets.toml, or in the Streamlit Cloud 'Secrets' UI
    for deployment). If no password is configured, the app runs unprotected."""
    if "APP_PASSWORD" not in st.secrets:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("⚽ Match GPS Dashboard")
    with st.form("login_form"):
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(pwd, st.secrets["APP_PASSWORD"]):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


REQUIRED_COLUMNS = [
    "Player Name",
    "Total Distance (m)",
    "High Speed Running (m)",
    "Sprint Distance (m)",
    "High Intensity Actions",
    "High Intensity Distance (m)",
    "Max Speed (km/h)",
    "Game Minutes (mins)",
]

METRIC_COLUMNS = REQUIRED_COLUMNS[1:]  # everything except Player Name
SUMMARY_METRICS = [m for m in METRIC_COLUMNS if m not in ("Max Speed (km/h)", "Game Minutes (mins)")]
PLAYER_PROGRESS_METRICS = [
    ("Game Minutes (mins)", "Match Minutes"),
    ("Total Distance (m)", "Total Distance"),
    ("High Speed Running (m)", "High Speed Running"),
    ("Sprint Distance (m)", "Sprint Distance"),
    ("High Intensity Actions", "High Intensity Actions"),
    ("High Intensity Distance (m)", "HMLD"),
]
IDENTIFIER_COLUMNS = ["Match Date", "Opposition"]  # optional, used to distinguish matches

METRIC_LABELS = {
    "Total Distance (m)": "Total Distance",
    "High Speed Running (m)": "High Speed Running",
    "Sprint Distance (m)": "Sprint Distance",
    "High Intensity Actions": "High Intensity Actions",
    "High Intensity Distance (m)": "High Intensity Distance",
    "Max Speed (km/h)": "Max Speed",
    "Game Minutes (mins)": "Game Minutes",
}

TEAM_COLOR = "#1f4e8c"
HIGHLIGHT_COLOR = "#e8532b"
GRID_COLOR = "#d9d9d9"
MATCH_LABEL_COL = "_match_label"
SORT_DATE_COL = "_sort_date"
MATCH_INFO_COLUMNS = ["Competition", "Home Team", "Away Team", "Home Team Score", "Away Team Score"]
BADGE_PALETTE = ["#1f4e8c", "#e8532b", "#2f9e44", "#7048e8", "#f08c00", "#0c8599", "#c2255c", "#495057"]
LOGOS_DIR = "logos"
LOGO_EXTENSIONS = [".png", ".jpg", ".jpeg", ".svg", ".webp"]


# --------------------------------------------------------------------------
# Helpers: data loading & validation
# --------------------------------------------------------------------------
def _parse_csv_bytes(raw_bytes):
    df = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(io.StringIO(raw_bytes.decode(encoding)))
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        # Last resort: decode leniently rather than fail outright
        df = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8", errors="replace")))
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]  # keep first occurrence of any repeated column name
    return df


def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    return _parse_csv_bytes(uploaded_file.read())


def _to_gsheet_csv_url(sheet_url):
    """Convert a Google Sheets share/edit URL into its CSV export URL.
    Works for links like .../spreadsheets/d/<ID>/edit#gid=<GID>."""
    sheet_url = sheet_url.strip()
    if "/export?format=csv" in sheet_url:
        return sheet_url
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        return None
    sheet_id = match.group(1)
    gid_match = re.search(r"[#&?]gid=([0-9]+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def load_csv_from_gsheet(sheet_url):
    """Fetch a Google Sheet as CSV. The sheet must be shared as 'Anyone with the
    link' (Viewer) — this uses the public export endpoint, not the Sheets API,
    so no credentials are needed."""
    csv_url = _to_gsheet_csv_url(sheet_url)
    if csv_url is None:
        st.error("That doesn't look like a valid Google Sheets link.")
        return None
    try:
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(
            f"Couldn't fetch that Google Sheet ({exc}). Make sure it's shared as "
            "'Anyone with the link' → Viewer."
        )
        return None
    if resp.headers.get("Content-Type", "").startswith("text/html"):
        st.error(
            "Google returned a login/permission page instead of CSV data. "
            "Make sure the sheet is shared as 'Anyone with the link' → Viewer."
        )
        return None
    return _parse_csv_bytes(resp.content)


def validate_columns(df, label="Uploaded file"):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(
            f"**{label}** is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)} "
            f"(plus optional 'Match Date' / 'Opposition' to enable match selection and trend views)."
        )
        return False
    return True


def coerce_numeric(df):
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def team_per90(df, metric):
    """Team total for `metric` divided by team total minutes, normalized to a 90-min game."""
    total_minutes = df["Game Minutes (mins)"].sum()
    if not total_minutes:
        return float("nan")
    return df[metric].sum() / total_minutes * 90


def parse_match_date(series):
    """Robust date parser: tries strict ISO (YYYY-MM-DD) first since it's
    unambiguous, then falls back to day-first parsing (DD/MM/YY etc.) for
    anything that didn't match. Avoids blanket dayfirst=True corrupting
    already-unambiguous ISO dates."""
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)
    return parsed


def _display_date(series):
    """Formats a date column as 'DD Mon YYYY' for display; falls back to the raw
    string if it can't be parsed as a date."""
    parsed = parse_match_date(series)
    formatted = parsed.dt.strftime("%d %b %Y")
    raw = series.astype(str).str.strip()
    return formatted.where(parsed.notna(), raw)


def build_match_labels(df):
    """
    Adds a MATCH_LABEL_COL identifying each unique match, built from whichever of
    Match Date / Opposition are present. If neither is present, the whole file is
    treated as a single match.
    """
    df = df.copy()
    cols_present = [c for c in IDENTIFIER_COLUMNS if c in df.columns]

    if not cols_present:
        df[MATCH_LABEL_COL] = "Uploaded Match"
        return df, cols_present

    if len(cols_present) == 2:
        date_part = _display_date(df["Match Date"])
        opp_part = df["Opposition"].astype(str).str.strip()
        df[MATCH_LABEL_COL] = date_part + " vs " + opp_part
    elif cols_present[0] == "Match Date":
        df[MATCH_LABEL_COL] = _display_date(df["Match Date"])
    else:
        df[MATCH_LABEL_COL] = df[cols_present[0]].astype(str).str.strip()

    return df, cols_present


def sorted_match_labels(df, cols_present, descending=True):
    """Unique match labels, ordered chronologically if Match Date is available."""
    subset_cols = [MATCH_LABEL_COL] + (["Match Date"] if "Match Date" in cols_present else [])
    subset = df[subset_cols].drop_duplicates()
    if "Match Date" in cols_present:
        subset[SORT_DATE_COL] = parse_match_date(subset["Match Date"])
        subset = subset.sort_values(SORT_DATE_COL, ascending=not descending)
    return subset[MATCH_LABEL_COL].tolist()


# --------------------------------------------------------------------------
# Helpers: matplotlib figures (PDF export only)
# --------------------------------------------------------------------------
def bar_chart_mpl(df, metric, highlight_player=None, title=None):
    data = df[["Player Name", metric]].dropna().sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(data))))
    colors = [HIGHLIGHT_COLOR if p == highlight_player else TEAM_COLOR for p in data["Player Name"]]
    ax.barh(data["Player Name"], data[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} by Player")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def trend_chart_mpl(hist_df, player, metric):
    pdata = hist_df[hist_df["Player Name"] == player].copy()
    if "Match Date" in pdata.columns:
        pdata["_sort_dt"] = parse_match_date(pdata["Match Date"])
        pdata = pdata.sort_values("_sort_dt")
        x = pdata["Match Date"]
    else:
        x = range(len(pdata))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, pdata[metric], marker="o", color=TEAM_COLOR, linewidth=2)
    if "Match Date" in hist_df.columns:
        hist_df = hist_df.copy()
        hist_df["_sort_dt"] = parse_match_date(hist_df["Match Date"])
        team_avg = hist_df.sort_values("_sort_dt").groupby("Match Date", sort=False)[metric].mean()
        team_avg = team_avg.reindex(pdata["Match Date"])
    else:
        team_avg = None
    if team_avg is not None:
        ax.plot(x, team_avg.values, linestyle="--", color="#999999", linewidth=1.5, label="Team Avg")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — Trend: {player}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def player_vs_avg_chart_mpl(current_row, team_avg):
    metrics = METRIC_COLUMNS
    player_vals = [current_row[m] for m in metrics]
    avg_vals = [team_avg[m] for m in metrics]
    labels = [METRIC_LABELS[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - width / 2, player_vals, width, label=current_row["Player Name"], color=HIGHLIGHT_COLOR)
    ax.bar(x + width / 2, avg_vals, width, label="Team Average", color=TEAM_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_title(f"{current_row['Player Name']} vs Team Average")
    fig.tight_layout()
    return fig


def df_to_table_figure(df, title, max_rows=30):
    show_df = df.head(max_rows).copy()
    for c in show_df.select_dtypes(include=[float]).columns:
        show_df[c] = show_df[c].round(1)

    fig_h = 0.4 * (len(show_df) + 1) + 0.6
    fig, ax = plt.subplots(figsize=(11, min(fig_h, 14)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14)
    tbl = ax.table(cellText=show_df.values, colLabels=show_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(TEAM_COLOR)
            cell.set_text_props(color="white", weight="bold")
    fig.tight_layout()
    return fig


def team_overview_table_figure(df, max_rows=30):
    cols = ["Player Name"] + METRIC_COLUMNS
    return df_to_table_figure(df[cols].sort_values("Game Minutes (mins)", ascending=False), "Team Overview — All Metrics", max_rows)


def build_matrix_chart(df, raw_df=None, window_labels=None, row_height=26, bar_width=170, sort_by="Game Minutes (mins)"):
    """
    Single Altair chart showing all 7 metrics as columns, one row per player.
    - Shared y-axis (player names) across all metric columns, labels shown once.
    - Independent x-scale and color scale per metric (units/ranges differ).
    - Bars colored light-to-dark blue by value (per metric's own min/max).
    - Data labels rounded to whole numbers, except Max Speed (1 decimal place).
    - Players sorted by `sort_by`, highest to lowest, by default.
    - If raw_df + window_labels (most-recent-first match labels) are given, the
      tooltip also shows that metric's value across each of those matches.
    """
    player_order = df.sort_values(sort_by, ascending=False)["Player Name"].tolist()
    metric_order = [METRIC_LABELS[m] for m in METRIC_COLUMNS]
    max_speed_label = METRIC_LABELS["Max Speed (km/h)"]

    long_df = df.melt(id_vars="Player Name", value_vars=METRIC_COLUMNS, var_name="Metric", value_name="Value")
    long_df["Metric"] = long_df["Metric"].map(METRIC_LABELS)
    long_df = long_df.dropna(subset=["Value"])
    is_speed = long_df["Metric"] == max_speed_label
    long_df["Label"] = np.where(
        is_speed,
        long_df["Value"].map(lambda v: f"{v:.1f}"),
        long_df["Value"].round(0).astype(int).astype(str),
    )

    tooltip_fields = [
        alt.Tooltip("Player Name:N", title="Player"),
        alt.Tooltip("Metric:N"),
        alt.Tooltip("Value:Q", title="Value", format=",.1f"),
    ]

    if raw_df is not None and window_labels and len(window_labels) > 1:
        hist = raw_df[raw_df["Player Name"].isin(long_df["Player Name"].unique()) & raw_df[MATCH_LABEL_COL].isin(window_labels)]
        hist_long = hist.melt(id_vars=["Player Name", MATCH_LABEL_COL], value_vars=METRIC_COLUMNS, var_name="MetricCol", value_name="V")
        hist_long["Metric"] = hist_long["MetricCol"].map(METRIC_LABELS)
        pivot = hist_long.pivot_table(index=["Player Name", "Metric"], columns=MATCH_LABEL_COL, values="V", aggfunc="first")
        pivot = pivot.reindex(columns=window_labels).reset_index()
        long_df = long_df.merge(pivot, on=["Player Name", "Metric"], how="left")

        for lbl in window_labels:
            is_speed_row = long_df["Metric"] == max_speed_label
            raw_vals = long_df[lbl]
            formatted = []
            for val, speed_row in zip(raw_vals, is_speed_row):
                if pd.isna(val):
                    formatted.append("—")
                elif speed_row:
                    formatted.append(f"{val:.1f}")
                else:
                    formatted.append(str(int(round(val))))
            long_df[lbl] = formatted
            tooltip_fields.append(alt.Tooltip(f"{lbl}:N", title=lbl))

    chart_height = max(240, row_height * df["Player Name"].nunique())

    base = alt.Chart(long_df).encode(
        y=alt.Y("Player Name:N", sort=player_order, title=None),
    )

    bars = base.mark_bar().encode(
        x=alt.X("Value:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=True)),
        color=alt.Color("Value:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=tooltip_fields,
    )

    labels = base.mark_text(align="left", baseline="middle", dx=4, fontSize=13, color="#333333").encode(
        x=alt.X("Value:Q"),
        text=alt.Text("Label:N"),
        tooltip=tooltip_fields,
    )

    layer = (bars + labels).properties(height=chart_height, width=bar_width)

    chart = (
        layer.facet(
            column=alt.Column(
                "Metric:N",
                sort=metric_order,
                title=None,
                header=alt.Header(labelFontWeight="bold", labelFontSize=12, labelPadding=6),
            )
        )
        .resolve_scale(x="independent", color="independent")
        .configure_view(strokeWidth=0)
    )
    return chart


# --------------------------------------------------------------------------
# Dynamic match header (Competition / Home vs Away scoreline)
# --------------------------------------------------------------------------
def _team_initials(name):
    if not isinstance(name, str) or not name.strip():
        return "?"
    parts = [p for p in name.replace("FC", "").split() if p]
    if not parts:
        return name[:2].upper()
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _team_badge_html(name, size=44):
    color = BADGE_PALETTE[zlib.crc32(str(name).encode()) % len(BADGE_PALETTE)]
    return (
        f"<div style='width:{size}px;height:{size}px;border-radius:50%;background:{color};"
        f"display:flex;align-items:center;justify-content:center;color:white;"
        f"font-weight:700;font-size:{size*0.38}px;margin:0 auto;'>{_team_initials(name)}</div>"
    )


def _find_team_logo(team_name):
    """Looks for logos/<Team Name>.<ext> (spaces, underscores, hyphens all tried)."""
    if not isinstance(team_name, str) or not team_name.strip():
        return None
    for candidate in (team_name, team_name.replace(" ", "_"), team_name.replace(" ", "-")):
        for ext in LOGO_EXTENSIONS:
            path = os.path.join(LOGOS_DIR, candidate + ext)
            if os.path.isfile(path):
                return path
    return None


def _render_team_badge(team_name, size=44):
    """Shows the team's logo file if found in logos/, else a colored-initials badge."""
    logo_path = _find_team_logo(team_name)
    if logo_path:
        st.image(logo_path, width=size)
    else:
        st.markdown(_team_badge_html(team_name, size=size), unsafe_allow_html=True)


def render_match_header(match_row):
    """Renders [Competition] then [Home logo][Home] score - score [Away][Away logo].
    Falls back to a static title if match-info columns aren't present."""
    has_info = all(c in match_row.index for c in MATCH_INFO_COLUMNS) and pd.notna(match_row.get("Home Team"))
    if not has_info:
        st.title("⚽ Match GPS Dashboard")
        return

    competition = match_row.get("Competition")
    home_team, away_team = str(match_row["Home Team"]), str(match_row["Away Team"])
    hs, as_ = match_row.get("Home Team Score"), match_row.get("Away Team Score")
    hs_str = "–" if pd.isna(hs) else str(int(hs))
    as_str = "–" if pd.isna(as_) else str(int(as_))

    if pd.notna(competition):
        st.markdown(
            f"<div style='text-align:center;color:#888;font-size:0.8rem;"
            f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;'>{competition}</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 3, 1, 0.6, 1, 3, 1])
    with c1:
        _render_team_badge(home_team)
    with c2:
        st.markdown(f"<div style='text-align:right;font-size:1.4rem;font-weight:700;padding-top:6px;'>{home_team}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align:center;font-size:1.8rem;font-weight:800;padding-top:2px;'>{hs_str}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div style='text-align:center;font-size:1.8rem;font-weight:800;padding-top:2px;color:#999;'>-</div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div style='text-align:center;font-size:1.8rem;font-weight:800;padding-top:2px;'>{as_str}</div>", unsafe_allow_html=True)
    with c6:
        st.markdown(f"<div style='text-align:left;font-size:1.4rem;font-weight:700;padding-top:6px;'>{away_team}</div>", unsafe_allow_html=True)
    with c7:
        _render_team_badge(away_team)


# --------------------------------------------------------------------------
# Sidebar — single upload & filters
# --------------------------------------------------------------------------
st.sidebar.title("⚽ GPS Dashboard")
st.sidebar.markdown("Load a single match-data source to get started.")

data_source = st.sidebar.radio("Data source", ["Upload CSV", "Google Sheet"], key="data_source", horizontal=True)

raw_df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Match Data CSV", type=["csv"], key="match_upload")
    if uploaded_file is not None:
        raw_df = load_csv(uploaded_file)
else:
    st.sidebar.caption("Sheet must be shared as 'Anyone with the link' → Viewer.")
    sheet_url = st.sidebar.text_input("Google Sheet URL", key="gsheet_url")
    if sheet_url:
        raw_df = load_csv_from_gsheet(sheet_url)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Required columns: Player Name, Total Distance (m), High Speed Running (m), "
    "Sprint Distance (m), High Intensity Actions, High Intensity Distance (m), "
    "Max Speed (km/h), Game Minutes (mins).\n\n"
    "Add optional 'Match Date' and/or 'Opposition' columns (one row per player per match) "
    "and the app will automatically detect multiple matches, letting you pick any single "
    "match to view and unlocking trend charts. A file with only one match is treated as "
    "a single-match upload."
)

with st.sidebar.expander("Don't have a file handy? Try the sample data"):
    st.caption("Two sample files included with this app (25 players, 30 matches).")
    try:
        with open("sample_data/full_season_data.csv", "rb") as f:
            st.download_button("Download full_season_data.csv (multi-match)", f, file_name="full_season_data.csv")
        with open("sample_data/single_match_data.csv", "rb") as f:
            st.download_button("Download single_match_data.csv (one match)", f, file_name="single_match_data.csv")
    except FileNotFoundError:
        st.caption("(Sample files not found next to app.py)")

if raw_df is None:
    st.title("⚽ Match GPS Dashboard")
    st.info(
        "👈 Upload a match-data CSV or link a Google Sheet in the sidebar to begin. "
        "Include optional Match Date / Opposition columns with multiple matches to unlock "
        "match selection and trend views — otherwise a single match works fine too."
    )
    st.stop()

if not validate_columns(raw_df):
    st.stop()
raw_df = coerce_numeric(raw_df)
raw_df, id_cols_present = build_match_labels(raw_df)

has_multiple_matches = raw_df[MATCH_LABEL_COL].nunique() > 1
match_order_desc = sorted_match_labels(raw_df, id_cols_present, descending=True)
most_recent_label = match_order_desc[0]
has_history = has_multiple_matches

# Sidebar filters + navigation
st.sidebar.markdown("---")
all_players = sorted(raw_df["Player Name"].dropna().unique().tolist())
player_filter = st.sidebar.multiselect("Filter players (Team Overview)", options=all_players, default=all_players)

st.sidebar.markdown("---")
page = st.sidebar.radio("View", ["Team Overview", "Player Profile", "Export PDF"])
st.sidebar.markdown("---")
save_pdf_clicked = st.sidebar.button("📄 Save Current View as PDF")

# --------------------------------------------------------------------------
# Header — dynamic, driven by a global match selector
# --------------------------------------------------------------------------
header_col, select_col = st.columns([3, 2])
with select_col:
    if has_multiple_matches:
        competition_lookup = (
            raw_df.groupby(MATCH_LABEL_COL)["Competition"].first().to_dict()
            if "Competition" in raw_df.columns
            else {}
        )

        def _format_match_option(label):
            comp = competition_lookup.get(label)
            if not np.isscalar(comp):
                return str(label)
            try:
                is_missing = pd.isna(comp)
            except (TypeError, ValueError):
                is_missing = comp is None
            if is_missing:
                return str(label)
            comp_str = str(comp).strip()
            return f"{label} ({comp_str})" if comp_str else str(label)

        selected_match_label = st.selectbox(
            "Match (Date + Opposition)",
            options=match_order_desc,
            format_func=_format_match_option,
            key="global_match_select",
        )
    else:
        selected_match_label = raw_df[MATCH_LABEL_COL].iloc[0]

selected_df = raw_df[raw_df[MATCH_LABEL_COL] == selected_match_label].copy()

with header_col:
    render_match_header(selected_df.iloc[0])

if save_pdf_clicked:
    pdf_current = selected_df[selected_df["Player Name"].isin(player_filter)]
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig1 = team_overview_table_figure(pdf_current)
        pdf.savefig(fig1)
        plt.close(fig1)
        fig2 = bar_chart_mpl(pdf_current, "Total Distance (m)", title="Total Distance — Team Overview")
        pdf.savefig(fig2)
        plt.close(fig2)
        sel_player = st.session_state.get("player_select")
        if sel_player and sel_player in selected_df["Player Name"].values:
            prow = selected_df[selected_df["Player Name"] == sel_player].iloc[0]
            tavg = selected_df[METRIC_COLUMNS].mean()
            fig3 = player_vs_avg_chart_mpl(prow, tavg)
            pdf.savefig(fig3)
            plt.close(fig3)
        d = pdf.infodict()
        d["Title"] = "Match GPS Dashboard Report"
        d["CreationDate"] = datetime.now()
    buffer.seek(0)
    st.sidebar.download_button("Download PDF", data=buffer, file_name="dashboard_view.pdf", mime="application/pdf")

if page != "Player Profile":
    st.subheader("Team Data Summary")
    summary_cols = st.columns(len(SUMMARY_METRICS))
    historical_all_df = raw_df[raw_df[MATCH_LABEL_COL] != selected_match_label] if has_history else None

    for col, metric in zip(summary_cols, SUMMARY_METRICS):
        selected_val = team_per90(selected_df, metric)
        delta_text = None
        if historical_all_df is not None and not historical_all_df.empty:
            hist_val = team_per90(historical_all_df, metric)
            if hist_val:
                diff = selected_val - hist_val
                pct = diff / hist_val * 100
                delta_text = f"{diff:+,.1f} ({pct:+.1f}%)"
        col.metric(
            METRIC_LABELS[metric],
            f"{selected_val:,.1f}",
            delta=delta_text,
            help=f"Team per-90 rate — {metric} (selected match vs. all other matches)",
        )

st.markdown(
    """
    <style>
    hr {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rrem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --------------------------------------------------------------------------
# Page content (selected via sidebar nav)
# --------------------------------------------------------------------------

# ---- Team Overview ---------------------------------------------------------
if page == "Team Overview":
    st.subheader("Team Overview")

    overview_df = selected_df[selected_df["Player Name"].isin(player_filter)]

    if overview_df.empty:
        st.warning("No players selected — adjust the filter in the sidebar.")
    else:
        if selected_match_label in match_order_desc:
            idx = match_order_desc.index(selected_match_label)
            window_labels = match_order_desc[idx: idx + 5]
        else:
            window_labels = [selected_match_label]
        matrix = build_matrix_chart(overview_df, raw_df=raw_df, window_labels=window_labels, sort_by="Game Minutes (mins)")
        st.altair_chart(matrix, use_container_width=True)

    st.caption("Sorted by Game Minutes, highest to lowest, by default. Hover a bar to see the 5 most recent matches.")

# ---- Player Profile ---------------------------------------------------------
elif page == "Player Profile":
    st.subheader("Player Profile")
    selected_player = st.selectbox("Select player", all_players, key="player_select")

    player_matches = selected_df[selected_df["Player Name"] == selected_player]
    if player_matches.empty:
        st.info("Player not involved in current selected match")
    else:
        player_row = player_matches.iloc[0]
        team_avg = selected_df[METRIC_COLUMNS].mean()

        pcols = st.columns(len(METRIC_COLUMNS))
        for col, metric in zip(pcols, METRIC_COLUMNS):
            delta = player_row[metric] - team_avg[metric]
            col.metric(METRIC_LABELS[metric], f"{player_row[metric]:,.1f}", delta=f"{delta:+.1f} vs avg")

        st.markdown("---")
        st.markdown(f"**{selected_player} — Historical Trend**")

        player_all = raw_df[raw_df["Player Name"] == selected_player].copy()
        if "Match Date" in player_all.columns:
            player_all["_plot_date"] = parse_match_date(player_all["Match Date"])
            player_all = player_all.sort_values("_plot_date")
            x_enc = alt.X("_plot_date:T", title=None)
        else:
            player_all = player_all.reset_index(drop=True)
            player_all["_plot_date"] = player_all.index
            x_enc = alt.X("_plot_date:O", title=None, axis=alt.Axis(labels=False))

        tooltip_fields = [alt.Tooltip("_plot_date:T", title="Date", format="%d %b %Y")] if "Match Date" in player_all.columns else []
        if "Opposition" in player_all.columns:
            tooltip_fields.append(alt.Tooltip("Opposition:N"))
        if "Competition" in player_all.columns:
            tooltip_fields.append(alt.Tooltip("Competition:N"))

        for metric, display_label in PLAYER_PROGRESS_METRICS:
            st.markdown(f"**{display_label}**")
            chart = (
                alt.Chart(player_all.dropna(subset=[metric]))
                .mark_area(color=TEAM_COLOR, opacity=0.45, line={"color": TEAM_COLOR})
                .encode(x=x_enc, y=alt.Y(f"{metric}:Q", title=None), tooltip=tooltip_fields + [alt.Tooltip(f"{metric}:Q", title=display_label, format=",.1f")])
                .properties(height=160)
            )
            st.altair_chart(chart, use_container_width=True)

# ---- Export PDF ---------------------------------------------------------
elif page == "Export PDF":
    st.subheader("Export Dashboard as PDF")
    st.caption(
        "Builds a shareable PDF snapshot: full team overview table + chart for a chosen "
        "metric (most recent match), plus a selected player's profile chart + table. "
        "Great for sending to players after a match."
    )

    export_metric = st.selectbox(
        "Team metric to chart", METRIC_COLUMNS, format_func=lambda m: METRIC_LABELS[m], key="export_metric"
    )
    export_player = st.selectbox("Player profile to include", all_players, key="export_player")
    include_trend = has_history and st.checkbox("Include trend chart for selected player", value=has_history)

    if st.button("Generate PDF", type="primary"):
        pdf_current = selected_df[selected_df["Player Name"].isin(player_filter)]
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            fig_team_table = team_overview_table_figure(pdf_current)
            pdf.savefig(fig_team_table)
            plt.close(fig_team_table)

            fig_team = bar_chart_mpl(pdf_current, export_metric, title=f"{METRIC_LABELS[export_metric]} — Team Overview")
            pdf.savefig(fig_team)
            plt.close(fig_team)

            prow = selected_df[selected_df["Player Name"] == export_player].iloc[0]
            tavg = selected_df[METRIC_COLUMNS].mean()
            fig_player = player_vs_avg_chart_mpl(prow, tavg)
            pdf.savefig(fig_player)
            plt.close(fig_player)

            fig_player_table = df_to_table_figure(prow.to_frame().T, title=f"{export_player} — Raw Match Data")
            pdf.savefig(fig_player_table)
            plt.close(fig_player_table)

            if include_trend:
                fig_trend = trend_chart_mpl(raw_df, export_player, export_metric)
                pdf.savefig(fig_trend)
                plt.close(fig_trend)

            d = pdf.infodict()
            d["Title"] = "Match GPS Dashboard Report"
            d["Author"] = "Match GPS Dashboard"
            d["CreationDate"] = datetime.now()

        buffer.seek(0)
        st.success("PDF generated.")
        st.download_button(
            "Download PDF",
            data=buffer,
            file_name=f"gps_report_{export_player.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )