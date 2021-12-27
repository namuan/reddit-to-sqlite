import logging
import sqlite3
import typing
from functools import partial
from itertools import takewhile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import praw  # type: ignore
import sqlite_utils
import typer

from .reddit_instance import get_auth, reddit_instance

LIMIT = 1000
SECONDS_IN_DAY = 60 * 60 * 24
logger = logging.getLogger(__name__)
app = typer.Typer()


def query_val(db: sqlite_utils.Database, qry: str) -> Optional[int]:
    "Safely get a single value using `qry`"
    try:
        curs = db.execute(qry)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return None
        raise
    result = curs.fetchone()
    logger.debug("qry=%s result=%s", qry, result)
    return result[0]


def latest_from_user_utc(
    db: sqlite_utils.Database, table_name: str, username: str
) -> Optional[int]:
    qry = f"select max(created_utc) from {table_name} where author = '{username}'"
    return query_val(db, qry)


def created_since(row: Any, target_sec_utc: Optional[int]) -> bool:
    result = (not target_sec_utc) or (row.created_utc >= target_sec_utc)
    logger.debug(
        "row.id=%s row.created_utc=%s >= target_sec_utc=%s? result",
        row.id,
        row.created_utc,
        target_sec_utc,
        result,
    )
    return result


def save_user(
    db: sqlite_utils.Database,
    reddit: Any,
    username: str,
    post_reload_sec: int,
    comment_reload_sec: int,
) -> None:
    user = reddit.redditor(username)
    latest_post_utc = latest_from_user_utc(db=db, table_name="posts", username=username)
    get_since = latest_post_utc and (latest_post_utc - post_reload_sec)
    logger.info("Getting posts by %s since timestamp %s", username, get_since)
    _takewhile = partial(created_since, target_sec_utc=get_since)

    assert isinstance(db["posts"], sqlite_utils.db.Table)
    db["posts"].upsert_all(
        (saveable(s) for s in takewhile(_takewhile, user.submissions.new(limit=LIMIT))),
        pk="id",
        alter=True,
    )

    get_since = latest_post_utc and (latest_post_utc - comment_reload_sec)
    logger.info("Getting comments by %s since timestamp %s", username, get_since)
    _takewhile = partial(created_since, target_sec_utc=get_since)

    assert isinstance(db["comments"], sqlite_utils.db.Table)
    db["comments"].upsert_all(
        (saveable(s) for s in takewhile(_takewhile, user.comments.new(limit=LIMIT))),
        pk="id",
        alter=True,
    )


def latest_post_in_subreddit_utc(
    db: sqlite_utils.Database, subreddit: str
) -> Optional[int]:
    qry = f"select max(created_utc) from posts where subreddit = '{subreddit}'"
    return query_val(db, qry)


def save_subreddit(
    db: sqlite_utils.Database,
    reddit: Any,
    subreddit_name: str,
    post_reload_sec: int,
    comment_reload_sec: int,
) -> None:
    subreddit = reddit.subreddit(subreddit_name)
    latest_post_utc = latest_post_in_subreddit_utc(db=db, subreddit=subreddit)
    get_since = latest_post_utc and (latest_post_utc - post_reload_sec)
    logger.info("Getting posts in %s since timestamp %s", subreddit, get_since)
    _takewhile = partial(created_since, target_sec_utc=get_since)
    for post in takewhile(_takewhile, subreddit.new(limit=LIMIT)):
        logger.debug("Post id %s", post.id)
        assert isinstance(db["posts"], sqlite_utils.db.Table)
        db["posts"].upsert(saveable(post), pk="id", alter=True)
        post.comments.replace_more()
        assert isinstance(db["comments"], sqlite_utils.db.Table)
        db["comments"].upsert_all(
            (saveable(c) for c in post.comments.list()),
            pk="id",
            alter=True,
        )


def legalize(val: Any) -> Any:
    """Convert `val` to a form that can be saved in sqlite"""

    if isinstance(val, praw.models.reddit.base.RedditBase):
        return str(val)
    return val


def _parent_ids_interpreted(dct: typing.Dict[str, typing.Any]) -> Dict[str, typing.Any]:
    if not dct.get("parent_id"):
        return dct

    prefix = dct["parent_id"][:3]
    dct["parent_clean_id"] = dct["parent_id"][3:]
    if prefix == "t1_":
        dct["parent_comment_id"] = dct["parent_clean_id"]
    elif prefix == "t3_":
        dct["parent_post_id"] = dct["parent_clean_id"]
    return dct


def saveable(item: Any) -> Dict[str, typing.Any]:
    """Generate a saveable dict from an instance"""

    result = {k: legalize(v) for k, v in item.__dict__.items() if not k.startswith("_")}
    return _parent_ids_interpreted(result)


def interpret_target(raw_target: str) -> Tuple[typing.Callable, str]:
    """Determine saving function and target string from input target"""

    help_message = "Target must be u/username or r/subreddit"
    savers = {"u": save_user, "r": save_subreddit}

    assert "/" in raw_target, help_message
    raw_target = raw_target.lower()
    pieces = raw_target.split("/")
    assert pieces[-2] in savers, help_message
    return savers[pieces[-2]], pieces[-1]


def create_index(db: sqlite_utils.Database, tbl: str, col: str) -> None:
    try:
        db[tbl].create_index([col], if_not_exists=True)  # type: ignore
    except sqlite3.OperationalError:
        logger.exception("Error indexing %s.%s:", tbl, col)


def create_fts_index(db: sqlite_utils.Database, tbl: str, cols: list) -> None:
    try:
        db[tbl].enable_fts(cols, tokenize="porter", create_triggers=True)
    except sqlite3.OperationalError:
        logger.exception("While setting up full-text search on %s.%s:", tbl, cols)


def setup_ddl(db: sqlite_utils.Database) -> None:
    for tbl in ("posts", "comments"):
        for col in ("author", "created_utc", "subreddit", "score", "removed"):
            create_index(db, tbl, col)
    for col in ("parent_clean_id", "parent_comment_id", "parent_post_id"):
        create_index(db, "comments", col)

    create_fts_index(db, "posts", ["title", "selftext"])
    create_fts_index(
        db,
        "comments",
        [
            "body",
        ],
    )


def set_loglevel(verbosity: int) -> None:
    verbosity = min(verbosity, 2)
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(log_levels[verbosity])
    logger.addHandler(logging.StreamHandler())


def load_data_and_save(
    auth: Path, target: str, db: Path, post_reload: int, comment_reload: int
) -> None:
    reddit = reddit_instance(get_auth(auth.expanduser()))
    saver, save_me = interpret_target(target)
    database = sqlite_utils.Database(db.expanduser())
    saver(
        database,
        reddit,
        save_me,
        post_reload_sec=post_reload * SECONDS_IN_DAY,
        comment_reload_sec=comment_reload * SECONDS_IN_DAY,
    ),
    item_view_def = (Path(__file__).parent / "view_def.sql").read_text()
    database.create_view("items", item_view_def, replace=True)
    setup_ddl(database)


@app.command()
def main(
    target: str = typer.Argument(str, help="u/username or r/subreddit to collect"),
    auth: Path = typer.Option(
        Path("~/.config/reddit-to-sqlite.json"),
        help="File to retrieve/save Reddit auth",
    ),
    db: Path = typer.Option(Path("reddit.db"), help="database file"),
    post_reload: int = typer.Option(7, help="Age of posts to reload (days)"),
    comment_reload: int = typer.Option(7, help="Age of posts to reload (days)"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="More logging"),
) -> None:
    """Load posts and comments from Reddit to sqlite."""
    set_loglevel(verbosity=verbose)
    load_data_and_save(auth, target, db, post_reload, comment_reload)


if __name__ == "__main__":
    app()
