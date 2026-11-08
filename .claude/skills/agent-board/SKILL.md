---
name: agent-board
description: Use when a team wants its coding agents (Claude Code, Codex, any CLI agent) to coordinate through GitHub without people relaying messages - a shared message board on a pinned issue where agents hand work to each other, tag teammates by GitHub login and wake when tagged - or when asked to install or set up the agent board in a repo.
---

# Agent board

One pinned, locked GitHub issue is the team's message board. Each teammate's agent posts handoffs there with `board.py`, tags the teammate it needs by GitHub login, and keeps a background `wait` that returns when someone else posts. `board.py` needs only `gh` and `uv`, and reads posts only from the logins listed in it, so a public repo is safe.

## Files in this skill

| File | Goes to |
|---|---|
| `board.py` | the repo, e.g. `tools/board.py` |
| `hooks.json` | merged into the repo's `.claude/settings.json` |
| `rules.md` | a section of the repo's AGENTS.md |
| `issue.md` | the board issue's body |
| `test_board.py` | stays here; run it after changing `board.py` |

## Installing in a repo

Install only when the user asks for it in that repo. It is a change to the repo: follow the repo's own branch, PR and ownership rules, and if they give scripts, CI or agent config to someone else, get that person's OK first.

1. **Check.** `gh repo view --json nameWithOwner,visibility,viewerPermission` must show WRITE, MAINTAIN or ADMIN. Look for an existing board: `gh issue list --search "Agent board in:title"`.
2. **Ask the user, in one message:**
   - each teammate's GitHub login and the folders they own (start from `gh api repos/{owner}/{repo}/collaborators --jq '.[].login'`, which leaves out pending invitations);
   - the team lead;
   - which agents the team uses;
   - where the script goes: the repo's existing `scripts/` or `tools/`, else `tools/board.py`.

   COMMAND is `uv run <path>`. uv is the only runner: `python3` is a Store stub on many Windows machines and below 3.11 on stock macOS.
3. **Open the board.** This posts on GitHub, so confirm with the user first. Fill `issue.md`, then run `gh issue create --title "Agent board" --body-file <filled copy>`, `gh issue pin <N>` and `gh issue lock <N>`.
4. **Copy.** Copy `board.py` to the chosen path and set `REPO`, `BOARD_ISSUE`, `COMMAND` and `TEAM_LOGINS` (every teammate's login) at the top. Add `.board/` to `.gitignore`. Check with `git check-ignore -v` that neither `.claude/settings.json` nor the script is ignored.
5. **Hooks.** Merge `hooks.json` into `.claude/settings.json`, swapping in the chosen path. If the file already has hooks, append to its `SessionStart` and `UserPromptSubmit` arrays; never replace what's there.
6. **Rules.** Fill every `{{...}}` in `rules.md` (one table row per owner) and add it as a section of AGENTS.md, creating the file if needed; Codex and most other agents read AGENTS.md. Claude Code reads CLAUDE.md instead, so make sure CLAUDE.md exists and contains the line `@AGENTS.md`.
7. **Prove it.**
   - `COMMAND whoami` prints the user's login.
   - With the user's OK, post `[<Name>'s Claude] board test` and check that `COMMAND read -n 3` shows it.
   - A teammate tags the user, from their own checkout or by commenting on the issue in a browser, and the user's background `wait` returns with `** MENTION **`.
8. **Ship.** Commit and open a PR the repo's usual way. Once merged, each teammate runs `gh auth login` once and restarts their agent.

Once it's installed, the hooks and the rules section drive the board, so agents in that repo don't need this skill. To add a teammate later, add their login to `TEAM_LOGINS` and the owner table.

## Changing board.py

Edit the copy in this folder, then from this folder run `uv run --no-project --with pytest pytest -q -p no:cacheprovider test_board.py`. Each repo keeps its own copy; to update a repo, copy it again and keep that repo's four constants.

## Common mistakes

| Symptom | Fix |
|---|---|
| A teammate's existing hooks vanished | `.claude/settings.json` was overwritten; merge into the arrays instead |
| A teammate's posts never reach anyone | Their login isn't in `TEAM_LOGINS` |
| `wait` exits with a `board:` error | A setup problem it won't retry: wrong `REPO`/`BOARD_ISSUE`, gh missing or logged out, or `TEAM_LOGINS` empty. Being offline is retried |
| Mentions go to the wrong person just after `gh auth switch` | The login cache refreshes within an hour; delete `.board/login` to refresh now |
| No board text at session start on Windows | Check `/hooks`. If `$CLAUDE_PROJECT_DIR` didn't expand, use `uv run tools/board.py session-start`, since hooks normally run from the project root |
| Two background waits fighting | One per session; start the next only after it returns |
