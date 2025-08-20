# local_index.py
import sqlite3, pandas as pd

DB_PATH = "keiba_local.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS my_index (
  date    INTEGER NOT NULL,
  jyocd   INTEGER NOT NULL,
  racenum INTEGER NOT NULL,
  umaban  INTEGER NOT NULL,
  ver     TEXT    NOT NULL DEFAULT 'v1',
  score   REAL,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (date,jyocd,racenum,umaban,ver)
);
"""

def init_db(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    con.commit()
    con.close()

def upsert_index(df: pd.DataFrame, ver: str = "v1", db_path=DB_PATH):
    need = ["date","jyocd","racenum","umaban","score"]
    assert set(need).issubset(df.columns), f"必要列不足: {need}"
    df = (df[need].dropna(subset=["score"])
                  .astype({"date":"int64","jyocd":"int64","racenum":"int64","umaban":"int64","score":"float"}))
    df["ver"] = ver
    con = sqlite3.connect(db_path)
    con.executemany(
        """INSERT INTO my_index(date,jyocd,racenum,umaban,ver,score)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(date,jyocd,racenum,umaban,ver)
           DO UPDATE SET score=excluded.score, updated_at=CURRENT_TIMESTAMP;""",
        df[["date","jyocd","racenum","umaban","ver","score"]].itertuples(index=False, name=None)
    )
    con.commit(); con.close()

def fetch_race(date:int, jyocd:int, racenum:int, ver:str="v1", db_path=DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    q = """SELECT umaban, score FROM my_index
           WHERE date=? AND jyocd=? AND racenum=? AND ver=?
           ORDER BY score DESC"""
    out = pd.read_sql_query(q, con, params=(int(date), int(jyocd), int(racenum), ver))
    con.close()
    return out

