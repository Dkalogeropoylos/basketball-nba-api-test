
import io
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA SportsDataverse Full Test", page_icon="🏀", layout="wide")
st.title("🏀 NBA SportsDataverse Full Test")
st.caption(
    "Standalone cloud-safe NBA data test. Reads processed SportsDataverse GitHub release assets. "
    "It does NOT call stats.nba.com and does NOT touch the WNBA production model."
)

REPO = "sportsdataverse/sportsdataverse-data"
TAGS = {
    "Player boxscores": "espn_nba_player_boxscores",
    "Team boxscores": "espn_nba_team_boxscores",
    "Play-by-play": "espn_nba_pbp",
    "Schedules": "espn_nba_schedules",
}
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "nba-sdv-test/1.0"}

@st.cache_data(ttl=3600, show_spinner=False)
def release_assets(tag):
    url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    return [{
        "name": a["name"],
        "url": a["browser_download_url"],
        "size": a.get("size", 0),
    } for a in js.get("assets", [])]

def choose_season_asset(assets, season):
    s = str(int(season))
    # Prefer parquet; accept both likely naming conventions.
    candidates = [a for a in assets if s in a["name"]]
    if not candidates:
        return None
    order = {".parquet": 0, ".csv": 1, ".rds": 2}
    def rank(a):
        nm = a["name"].lower()
        ext_rank = 9
        for ext, rr in order.items():
            if nm.endswith(ext):
                ext_rank = rr
                break
        # Prefer canonical full dataset files, not manifests/timestamps.
        penalty = 1 if ("manifest" in nm or "repo" in nm or "timestamp" in nm) else 0
        return (penalty, ext_rank, len(nm))
    return sorted(candidates, key=rank)[0]

@st.cache_data(ttl=3600, show_spinner=False)
def load_asset(url, name):
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    raw = r.content
    lower = name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    raise ValueError(f"Unsupported asset format for cloud test: {name}")

def find_col(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def completeness_summary(label, df):
    out = {"Dataset": label, "Rows": len(df), "Columns": len(df.columns)}
    gid = find_col(df, ["game_id", "GAME_ID", "id"])
    if gid:
        out["Games"] = int(df[gid].astype(str).nunique())
    date = find_col(df, ["game_date", "GAME_DATE", "game_date_time", "date"])
    if date:
        d = pd.to_datetime(df[date], errors="coerce")
        if d.notna().any():
            out["From"] = str(d.min().date())
            out["Through"] = str(d.max().date())
    team = find_col(df, ["team_abbreviation", "TEAM_ABBR", "team_short_display_name"])
    if team:
        out["Teams"] = int(df[team].astype(str).nunique())
    player = find_col(df, ["athlete_id", "PLAYER_ID", "player_id"])
    if player:
        out["Players"] = int(df[player].astype(str).nunique())
    return out

season = st.number_input(
    "Season end year (2026 = 2025-26)",
    min_value=2002, max_value=2030, value=2026, step=1
)

st.info(
    "For the V2 core we need at minimum: team boxscores + player boxscores. "
    "PBP and schedule are extra layers for lineup/on-court and validation."
)

if st.button("Check SportsDataverse NBA releases", type="primary"):
    found = []
    with st.spinner("Reading GitHub release metadata..."):
        for label, tag in TAGS.items():
            try:
                assets = release_assets(tag)
                chosen = choose_season_asset(assets, season)
                if chosen:
                    found.append({
                        "Dataset": label,
                        "Tag": tag,
                        "Asset": chosen["name"],
                        "MB": round(chosen["size"]/1024/1024, 2),
                        "URL": chosen["url"],
                        "Status": "FOUND",
                    })
                else:
                    found.append({
                        "Dataset": label, "Tag": tag, "Asset": None, "MB": None,
                        "URL": None, "Status": f"NO {int(season)} ASSET"
                    })
            except Exception as exc:
                found.append({
                    "Dataset": label, "Tag": tag, "Asset": None, "MB": None,
                    "URL": None, "Status": f"ERROR: {exc}"
                })
    st.session_state["sdv_assets"] = pd.DataFrame(found)

assets_df = st.session_state.get("sdv_assets")
if isinstance(assets_df, pd.DataFrame) and not assets_df.empty:
    st.subheader("Release audit")
    st.dataframe(assets_df.drop(columns=["URL"]), use_container_width=True, hide_index=True)

    labels = assets_df.loc[assets_df["Status"].eq("FOUND"), "Dataset"].tolist()
    selected = st.multiselect(
        "Datasets to load",
        labels,
        default=[x for x in ["Team boxscores", "Player boxscores"] if x in labels]
    )

    if st.button("Load selected datasets"):
        summaries = []
        with st.spinner("Downloading processed release assets from GitHub..."):
            for label in selected:
                row = assets_df[assets_df["Dataset"].eq(label)].iloc[0]
                try:
                    df = load_asset(row["URL"], row["Asset"])
                    st.session_state[f"df_{label}"] = df
                    summaries.append(completeness_summary(label, df))
                except Exception as exc:
                    st.error(f"{label} failed: {exc}")
        if summaries:
            st.session_state["sdv_summaries"] = pd.DataFrame(summaries)

summaries = st.session_state.get("sdv_summaries")
if isinstance(summaries, pd.DataFrame):
    st.subheader("Completeness summary")
    st.dataframe(summaries, use_container_width=True, hide_index=True)

for label in TAGS.keys():
    df = st.session_state.get(f"df_{label}")
    if isinstance(df, pd.DataFrame):
        st.subheader(label)
        st.write(f"{len(df):,} rows × {len(df.columns):,} columns")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            f"Download {label} CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"nba_{label.lower().replace(' ','_').replace('-','_')}_{int(season)}.csv",
            mime="text/csv",
            key=f"dl_{label}",
        )

st.markdown("---")
st.caption(
    "Important: this is a data-source/completeness test only. "
    "No WNBA coefficients, models or files are imported."
)
