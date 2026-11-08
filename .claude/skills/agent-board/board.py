# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Agent board: one pinned GitHub issue where each teammate's coding agent posts handoffs and waits for its own tag.

Needs `gh` (logged in) and uv; `uv run` reads the inline metadata above and never syncs the host project's environment.
"""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# Set these when installing the board in a repo.
REPO = "owner/name"
BOARD_ISSUE = 1
COMMAND = "uv run tools/board.py"
# Only these logins can steer an agent: GitHub's own labels also let in read-only collaborators and org members.
TEAM_LOGINS: frozenset[str] = frozenset()

PAGE = 100
FIRST_CHECK = 10
GH_TIMEOUT = 60
LOGIN_TTL = 3600
# Claude Code shows only a short preview of hook output past 10,000 characters and never asks Claude to read the rest.
HOOK_BUDGET = 8000
POST_CLIP = 1500
PREFIX = re.compile(r"\[[^\]\n]+'s [^\]\n]+\]")
WAIT_RULE = (
    f"Keep ONE background `{COMMAND} wait` running (Bash, run_in_background); if one is already running, leave it. "
    "When it returns, handle its output: act on and reply to a MENTION; change course on other posts only if they "
    "touch your area. Then start the wait again. If it exits with a `board:` error, tell your human instead."
)


class BoardError(Exception):
    """gh is offline or GitHub answered unexpectedly; worth retrying."""


class SetupError(BoardError):
    """Retrying won't help until a person fixes something: gh missing or logged out, a wrong REPO or issue, no team."""


@dataclass(frozen=True)
class Comment:
    id: int
    login: str
    created_at: datetime
    body: str


class GitHub(Protocol):
    def login(self) -> str: ...
    def since(self, at: datetime) -> list[Comment]: ...
    def last(self, n: int) -> list[Comment]: ...
    def post(self, body: str) -> str: ...


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    # `.git` is a file inside a worktree, so each worktree keeps its own cursor and every agent sees every post.
    return next((p for p in (here, *here.parents) if (p / ".git").exists()), here)


STATE = repo_root() / ".board"
CURSOR = STATE / "cursor.json"
LOGIN = STATE / "login"


def team() -> frozenset[str]:
    if not TEAM_LOGINS:
        raise SetupError("TEAM_LOGINS is empty: list the team's GitHub logins at the top of the board script")
    return frozenset(login.lower() for login in TEAM_LOGINS)


def stamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str) -> datetime:
    try:
        at = datetime.fromisoformat(text)
    except ValueError:
        raise BoardError(f"--since needs an ISO time such as 2026-11-08T01:00:00Z, not {text!r}") from None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def parse_comment(raw: Any) -> Comment:
    """One comment as GitHub's REST API returns it; anything else is a BoardError, never a half-filled Comment."""
    try:
        comment = Comment(
            id=raw["id"],
            login=raw["user"]["login"],
            created_at=datetime.fromisoformat(raw["created_at"]),
            body=raw["body"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise BoardError(f"GitHub sent an unexpected comment ({e!r})") from None
    if not isinstance(comment.id, int) or not isinstance(comment.login, str) or not isinstance(comment.body, str):
        raise BoardError("GitHub sent an unexpected comment (wrong field types)")
    return comment


def parse_comments(raw: Any) -> list[Comment]:
    if not isinstance(raw, list):
        raise BoardError("GitHub sent something other than a list of comments")
    return [parse_comment(item) for item in raw]


def parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise BoardError(f"GitHub sent invalid JSON ({e})") from None


def mentions(body: str, login: str) -> bool:
    return re.search(rf"(?<![\w-])@({re.escape(login)}|all)(?![\w-])", body, re.IGNORECASE) is not None


def read_cursor(path: Path) -> tuple[int, datetime] | None:
    try:
        raw = json.loads(path.read_text())
        return int(raw["id"]), datetime.fromisoformat(raw["created_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None  # a missing, torn or hand-edited cursor restarts from the last few posts instead of wedging hooks


def write_cursor(path: Path, newest: Comment) -> None:
    scratch = path.with_name(f"{path.name}.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(json.dumps({"id": newest.id, "created_at": newest.created_at.isoformat()}))
        for attempt in range(5):
            try:
                scratch.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1)  # Windows refuses while another process (the other hook, OneDrive, antivirus) holds it
    except OSError as e:
        with contextlib.suppress(OSError):
            scratch.unlink(missing_ok=True)
        raise BoardError(f"couldn't save {path} ({e}); these posts show again at the next check") from None


def unread(gh: GitHub, me: str, cursor: Path) -> list[Comment]:
    """Teammates' posts since the last check, oldest first; the cursor moves past everything fetched."""
    members = team()
    seen = read_cursor(cursor)
    if seen is None:
        fetched = gh.last(FIRST_CHECK)
    else:
        seen_id, seen_at = seen
        # GitHub's `since` is inclusive and matches edits too, so posts already seen come back and are dropped here.
        fetched = [c for c in gh.since(seen_at) if c.id > seen_id]
    if not fetched:
        return []
    write_cursor(cursor, max(fetched, key=lambda c: c.id))
    mine = me.lower()
    return sorted((c for c in fetched if c.login.lower() in members and c.login.lower() != mine), key=lambda c: c.id)


def render(comments: list[Comment], me: str, clip: int = 0) -> str:
    blocks: list[str] = []
    for c in comments:
        flag = " ** MENTION **" if mentions(c.body, me) else ""
        body = c.body.replace("\r", "")
        if clip and len(body) > clip:
            full = f"`{COMMAND} read --since {stamp(c.created_at)}`"
            body = f"{body[:clip]}\n[cut at {clip} characters; {full} shows it in full]"
        # Indented, so no line of a post can pass for another post's header.
        blocks.append(f"--- @{c.login} {c.created_at:%Y-%m-%d %H:%M}Z (#{c.id}){flag}\n{textwrap.indent(body, '  ')}")
    return "\n".join(blocks)


def brief(comments: list[Comment], me: str) -> str:
    """The posts that fit a hook's output budget, tags first and then newest; says how to see the rest."""
    shown: list[Comment] = []
    used = 0
    for c in sorted(comments, key=lambda c: (not mentions(c.body, me), -c.id)):
        size = len(render([c], me, POST_CLIP)) + 1
        if used + size <= HOOK_BUDGET:
            shown.append(c)
            used += size
    text = render(sorted(shown, key=lambda c: c.id), me, POST_CLIP)
    if hidden := [c for c in comments if c not in shown]:
        oldest = min(hidden, key=lambda c: c.id)
        text += f"\n({len(hidden)} more not shown here: `{COMMAND} read --since {stamp(oldest.created_at)}` lists them)"
    return text


def check_post(text: str) -> str:
    text = text.strip()
    if not PREFIX.match(text):
        raise BoardError("a post starts with whose agent it is, e.g. [Sam's Claude]")
    return text


def wait(
    gh: GitHub, me: str, cursor: Path, interval: float, timeout: float, sleep: Callable[[float], None] = time.sleep
) -> list[Comment]:
    """Blocks until a teammate posts and returns the posts; [] once `timeout` seconds pass (0 waits forever).

    A SetupError raises at once, so a wrong REPO or a logged-out gh surfaces instead of hanging; other errors retry.
    """
    start = time.monotonic()
    while True:
        try:
            posts = unread(gh, me, cursor)
        except SetupError:
            raise
        except BoardError:
            posts = []  # offline or a GitHub hiccup: tried again on the next tick
        if posts:
            return posts
        if timeout and time.monotonic() - start >= timeout:
            return []
        sleep(interval)


def session_start(gh: GitHub, cursor: Path) -> str:
    try:
        me = gh.login()
        posts = brief(unread(gh, me, cursor), me) or "No unread posts."
    except Exception as e:  # a hook must never break the session, whatever went wrong
        posts = f"Board check failed ({e}). Run `{COMMAND} unread` to retry."
    return f"Agent board: {REPO} issue #{BOARD_ISSUE}; rules in AGENTS.md.\n{WAIT_RULE}\n{posts}"


def on_prompt(gh: GitHub, cursor: Path) -> str:
    try:
        me = gh.login()
        posts = unread(gh, me, cursor)
    except Exception:  # a hook must never break the session; session start already said how to check by hand
        return ""
    return f"New board posts (issue #{BOARD_ISSUE}):\n{brief(posts, me)}" if posts else ""


class GhCli:
    """The board through `gh api`, so each teammate's existing `gh auth login` is the only credential."""

    path = f"repos/{REPO}/issues/{BOARD_ISSUE}"

    def _api(self, *args: str, stdin: str | None = None) -> str:
        try:
            done = subprocess.run(
                ["gh", "api", *args],
                input=stdin,
                capture_output=True,
                encoding="utf-8",
                timeout=GH_TIMEOUT,
                check=False,
            )
        except FileNotFoundError:
            raise SetupError("gh not found: install it from https://cli.github.com, then run gh auth login") from None
        except (OSError, subprocess.TimeoutExpired) as e:
            raise BoardError(f"gh api failed ({e})") from None
        if done.returncode != 0:
            error = done.stderr.strip() or f"gh api exited {done.returncode}"
            status = re.search(r"\(HTTP (4\d\d)\)", error)
            # gh exits 4 when logged out, and a 4xx other than throttling won't clear by retrying.
            if done.returncode == 4 or (
                status and status[1] not in ("408", "429") and "rate limit" not in error.lower()
            ):
                raise SetupError(error)
            raise BoardError(error)
        return done.stdout

    def login(self) -> str:
        with contextlib.suppress(OSError):
            # An hour's cache saves a GitHub call on every prompt and still follows a `gh auth switch` soon after.
            fresh = time.time() - LOGIN.stat().st_mtime < LOGIN_TTL
            if fresh and (cached := LOGIN.read_text(encoding="utf-8").strip()):
                return cached
        me = self._api("user", "--jq", ".login").strip()
        with contextlib.suppress(OSError):
            STATE.mkdir(parents=True, exist_ok=True)
            LOGIN.write_text(me, encoding="utf-8")
        return me

    def since(self, at: datetime) -> list[Comment]:
        found: list[Comment] = []
        page = 1
        while True:
            batch = parse_comments(
                parse_json(self._api(f"{self.path}/comments?per_page={PAGE}&since={stamp(at)}&page={page}"))
            )
            found += batch
            if len(batch) < PAGE:
                return found
            page += 1

    def last(self, n: int) -> list[Comment]:
        issue = parse_json(self._api(self.path))
        count = issue.get("comments") if isinstance(issue, dict) else None
        if not isinstance(count, int):
            raise SetupError(f"{self.path} is not an issue")
        found: list[Comment] = []
        page = -(-count // PAGE)
        while page >= 1 and len(found) < n:
            found = parse_comments(parse_json(self._api(f"{self.path}/comments?per_page={PAGE}&page={page}"))) + found
            page -= 1
        return found[-n:] if n > 0 else []

    def post(self, body: str) -> str:
        # JSON on stdin: no shell quoting, and no Windows command-line length limit.
        reply = self._api(
            f"{self.path}/comments",
            "--method",
            "POST",
            "--input",
            "-",
            "--jq",
            ".html_url",
            stdin=json.dumps({"body": body}),
        )
        return reply.strip()


def main(argv: list[str] | None = None, gh: GitHub | None = None, cursor: Path = CURSOR) -> int:
    parser = argparse.ArgumentParser(prog=COMMAND, description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("post", help="post TEXT or stdin, starting [<name>'s <agent>]").add_argument("text", nargs="*")
    reading = sub.add_parser("read", help="the last N posts, or every post since a time")
    reading.add_argument("-n", type=int, default=10)
    reading.add_argument("--since", help="an ISO time such as 2026-11-08T01:00:00Z; overrides -n")
    sub.add_parser("unread", help="teammates' posts since your last check, then mark them read")
    waiting = sub.add_parser("wait", help="block until a teammate posts and print it; exit 124 after --timeout")
    waiting.add_argument("--interval", type=float, default=30)
    waiting.add_argument("--timeout", type=float, default=3600, help="seconds; 0 waits forever")
    sub.add_parser("whoami", help="your GitHub login")
    sub.add_parser("session-start", help="Claude Code SessionStart hook; never fails")
    sub.add_parser("prompt", help="Claude Code UserPromptSubmit hook; never fails")
    args = parser.parse_args(argv)
    gh = gh or GhCli()
    if args.command == "session-start":
        print(session_start(gh, cursor))
        return 0
    if args.command == "prompt":
        if out := on_prompt(gh, cursor):
            print(out)
        return 0
    try:
        match args.command:
            case "post":
                print(gh.post(check_post(" ".join(args.text) or sys.stdin.read())))
            case "whoami":
                print(gh.login())
            case "read":
                me, members = gh.login(), team()
                if args.since:
                    at = parse_time(args.since)
                    # `since` also matches older posts edited later; only posts made since then are wanted.
                    posts = [c for c in gh.since(at) if c.created_at >= at]
                else:
                    posts = gh.last(args.n)
                if out := render([c for c in posts if c.login.lower() in members], me):
                    print(out)
            case "unread":
                me = gh.login()
                if posts := unread(gh, me, cursor):
                    print(render(posts, me))
            case "wait":
                me = gh.login()
                posts = wait(gh, me, cursor, args.interval, args.timeout)
                if not posts:
                    print("No new posts before the timeout. Start the wait again.")
                    return 124
                print(render(posts, me))
    except BoardError as e:
        print(f"board: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")  # Windows pipes default to the locale codepage
    sys.exit(main())
