# traders/

Personal trading configurations per team member. Each subdirectory contains
that person's bot configs, launcher scripts, and LaunchAgent plists.

**Your stuff stays in your folder. Don't touch other people's folders.**

## Structure

```
traders/
  sai/          # Sai's 4-bot setup (2x Schwab paper, Tradier live, Tradier paper)
  bart/         # Vamshi (Bart) — add your configs here
  matt/         # Matt — add your configs here
```

## What goes here

- `.env.*` files with your broker credentials
- `launch_all_bots.sh` launcher script (with YOUR paths)
- `com.zdom.*.plist` LaunchAgent files (with YOUR paths)
- Any personal bot wrapper scripts

## What does NOT go here

- The core bot code (`scripts/zero_dte_bot.py`) — that's shared
- ML models, data pipelines, feature engineering — that's shared
- Anything in `config/settings.py` — that's shared defaults

## Setup for new team members

1. Copy `sai/` as a template: `cp -r traders/sai traders/yourname`
2. Edit all `.env.*` files with your broker credentials
3. Edit `launch_all_bots.sh` — update `PROJECT_DIR` to your path
4. Edit plist files — update paths to match your machine
5. Load plists: `launchctl load ~/Library/LaunchAgents/com.zdom.*.plist`
