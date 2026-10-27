"""Read and write Postgres with SQLAlchemy.

Computation uses wide frames (index=date, columns=ticker). Postgres stores
long rows (ticker, date, value). The helpers here convert between the two.
"""

import json
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from backtester.config import ROOT, database_url

FEATURES_SQL = ROOT / "sql" / "features.sql"

metadata = sa.MetaData()

tickers_table = sa.Table(
    "tickers",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("name", sa.Text),
    sa.Column("sector", sa.Text),
)

prices_table = sa.Table(
    "prices",
    metadata,
    sa.Column("ticker", sa.Text, primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("adj_close", sa.Numeric(14, 6)),
    sa.Column("volume", sa.BigInteger),
)

backtest_runs_table = sa.Table(
    "backtest_runs",
    metadata,
    sa.Column("run_id", sa.Integer, primary_key=True),
    sa.Column("signal_name", sa.Text),
    sa.Column("config_json", sa.dialects.postgresql.JSONB),
    sa.Column("sharpe", sa.Numeric),
    sa.Column("ann_return", sa.Numeric),
    sa.Column("max_dd", sa.Numeric),
)


def get_engine(url: str | None = None) -> sa.Engine:
    """Create a SQLAlchemy engine (defaults to the docker-compose Postgres)."""
    return sa.create_engine(url or database_url())


# ---------- wide <-> long ----------


def wide_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide (index=date, columns=ticker) -> long rows (ticker, date, value). Drops NaNs."""
    long = wide.copy()
    long.index.name = "date"
    long.columns.name = "ticker"
    long = long.reset_index().melt(id_vars="date", var_name="ticker", value_name=value_name)
    long = long.dropna(subset=[value_name])
    long = long[["ticker", "date", value_name]]
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def long_to_wide(long: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Long rows (ticker, date, value) -> wide (index=date, columns=ticker), sorted."""
    wide = long.pivot(index="date", columns="ticker", values=value_name)
    # Use one datetime unit everywhere so frames from different sources line up.
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index)).as_unit("ns")
    # SQLAlchemy hands back the column name as a quoted_name object; use a plain str.
    wide.index.name = "date"
    wide = wide.sort_index()
    wide = wide[sorted(wide.columns)]
    wide.columns.name = None
    return wide.astype(float)


def prices_to_long(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Combine wide prices and volume into long rows for the prices table.

    One row per (ticker, date) with a valid close. Volume can be missing.
    """
    long_prices = wide_to_long(prices, "adj_close")
    long_volume = wide_to_long(volume.reindex_like(prices), "volume")
    long = long_prices.merge(long_volume, on=["ticker", "date"], how="left")
    return long


# ---------- writes ----------


def upsert_tickers(engine: sa.Engine, tickers: pd.DataFrame) -> int:
    """Insert or update rows in tickers. Needs columns ticker, name, sector."""
    rows = []
    for row in tickers.itertuples(index=False):
        rows.append(
            {
                "ticker": row.ticker,
                "name": None if pd.isna(row.name) else str(row.name),
                "sector": None if pd.isna(row.sector) else str(row.sector),
            }
        )
    if not rows:
        return 0

    stmt = insert(tickers_table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={"name": stmt.excluded.name, "sector": stmt.excluded.sector},
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)
    return len(rows)


def upsert_prices(engine: sa.Engine, long: pd.DataFrame, chunk_size: int = 20_000) -> int:
    """Insert or update rows in prices from long rows (ticker, date, adj_close, volume)."""
    rows = []
    for row in long.itertuples(index=False):
        rows.append(
            {
                "ticker": row.ticker,
                "date": pd.Timestamp(row.date).date(),
                "adj_close": round(float(row.adj_close), 6),
                "volume": None if pd.isna(row.volume) else int(row.volume),
            }
        )

    stmt = insert(prices_table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "date"],
        set_={"adj_close": stmt.excluded.adj_close, "volume": stmt.excluded.volume},
    )
    with engine.begin() as conn:
        for start in range(0, len(rows), chunk_size):
            conn.execute(stmt, rows[start : start + chunk_size])
    return len(rows)


def delete_tickers(engine: sa.Engine, tickers: list[str]) -> None:
    """Delete these tickers from prices and tickers (used to clean up tests)."""
    with engine.begin() as conn:
        conn.execute(prices_table.delete().where(prices_table.c.ticker.in_(tickers)))
        conn.execute(tickers_table.delete().where(tickers_table.c.ticker.in_(tickers)))


def run_sql_file(engine: sa.Engine, path: Path = FEATURES_SQL) -> None:
    """Run a .sql file (for example, create the daily_features view)."""
    sql = Path(path).read_text()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def log_backtest_run(
    engine: sa.Engine,
    signal_name: str,
    config: dict,
    sharpe: float | None = None,
    ann_return: float | None = None,
    max_dd: float | None = None,
) -> int:
    """Record one backtest run with its config in backtest_runs. Returns run_id."""
    stmt = (
        backtest_runs_table.insert()
        .values(
            signal_name=signal_name,
            config_json=json.loads(json.dumps(config, default=str)),
            sharpe=sharpe,
            ann_return=ann_return,
            max_dd=max_dd,
        )
        .returning(backtest_runs_table.c.run_id)
    )
    with engine.begin() as conn:
        run_id = conn.execute(stmt).scalar_one()
    return int(run_id)


# ---------- reads ----------


def read_prices(
    engine: sa.Engine,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Read prices back as a wide frame (index=date, columns=ticker).

    column is "adj_close" or "volume". start and end are inclusive.
    """
    query = sa.select(prices_table.c.ticker, prices_table.c.date, prices_table.c[column])
    if tickers is not None:
        query = query.where(prices_table.c.ticker.in_(tickers))
    if start is not None:
        query = query.where(prices_table.c.date >= pd.Timestamp(start).date())
    if end is not None:
        query = query.where(prices_table.c.date <= pd.Timestamp(end).date())

    with engine.connect() as conn:
        long = pd.read_sql(query, conn)
    return long_to_wide(long, column)


def read_features(engine: sa.Engine, tickers: list[str] | None = None) -> pd.DataFrame:
    """Read the daily_features view as long rows (ticker, date, ...)."""
    sql = "SELECT * FROM daily_features"
    params = {}
    if tickers is not None:
        sql += " WHERE ticker IN :tickers"
        params["tickers"] = list(tickers)
    sql += " ORDER BY ticker, date"

    query = sa.text(sql)
    if tickers is not None:
        query = query.bindparams(sa.bindparam("tickers", expanding=True))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["adj_close", "ret_1d", "ret_5d", "mom_12_1"]:
        df[col] = df[col].astype(float)
    return df
