#!/usr/bin/env python3
"""
Bot Control Interface - Change v6 thresholds on-the-fly via Telegram

Usage:
    python scripts/bot_control.py status
    python scripts/bot_control.py set-threshold 0.60
    python scripts/bot_control.py set-consensus 2
    python scripts/bot_control.py force-entry "Manual override at 10:35 AM"
"""
import sys
import os
from pathlib import Path
from datetime import datetime

CONFIG_FILE = Path(__file__).parent.parent / ".env"
OVERRIDE_FILE = Path(__file__).parent.parent / ".bot_overrides.json"

def load_current_config():
    """Read current .env settings"""
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, val = line.strip().split('=', 1)
                    config[key] = val
    return config

def update_env(key, value):
    """Update .env file with new value"""
    lines = []
    found = False
    
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    
    if not found:
        lines.append(f"{key}={value}\n")
    
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(lines)

def save_override(override_type, value, reason=""):
    """Save override to JSON file (bot will check this)"""
    import json
    overrides = {}
    
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE, 'r') as f:
            overrides = json.load(f)
    
    overrides[override_type] = {
        'value': value,
        'timestamp': datetime.now().isoformat(),
        'reason': reason
    }
    
    with open(OVERRIDE_FILE, 'w') as f:
        json.dump(overrides, f, indent=2)

def cmd_status():
    """Show current settings"""
    config = load_current_config()
    
    print("=" * 60)
    print("v6 BOT CONTROL - CURRENT SETTINGS")
    print("=" * 60)
    print()
    print("Entry Thresholds:")
    print(f"  TP Ensemble:  ≥ {config.get('BOT_TP_ENTRY_THRESHOLD', '0.65')}")
    print(f"  Consensus:    ≥ {config.get('BOT_V6_CONSENSUS_MIN', '3')}/4 models")
    print()
    print("Entry Criteria (ALL must be true):")
    print(f"  ✓ TP25 ensemble ≥ {config.get('BOT_TP_ENTRY_THRESHOLD', '0.65')}")
    print(f"  ✓ TP50 ensemble ≥ {config.get('BOT_TP_ENTRY_THRESHOLD', '0.65')}")
    print(f"  ✓ TP25 consensus ≥ {config.get('BOT_V6_CONSENSUS_MIN', '3')}/4")
    print(f"  ✓ TP50 consensus ≥ {config.get('BOT_V6_CONSENSUS_MIN', '3')}/4")
    print()
    print("Quick Changes:")
    print("  - Lower threshold → More entries (less selective)")
    print("  - Higher threshold → Fewer entries (more selective)")
    print("  - Lower consensus → More entries (allow disagreement)")
    print("  - Higher consensus → Fewer entries (require agreement)")
    print()

def cmd_set_threshold(new_threshold):
    """Update entry threshold"""
    try:
        threshold = float(new_threshold)
        if threshold < 0 or threshold > 1:
            print("❌ Threshold must be between 0 and 1")
            return False
        
        update_env('BOT_TP_ENTRY_THRESHOLD', str(threshold))
        print(f"✅ Updated threshold: {threshold}")
        print(f"   Entries now need ensemble ≥ {threshold}")
        print()
        print("⚠️  Bot must be restarted for changes to take effect")
        return True
    except ValueError:
        print(f"❌ Invalid threshold: {new_threshold}")
        return False

def cmd_set_consensus(new_consensus):
    """Update consensus minimum"""
    try:
        consensus = int(new_consensus)
        if consensus < 0 or consensus > 4:
            print("❌ Consensus must be 0-4")
            return False
        
        update_env('BOT_V6_CONSENSUS_MIN', str(consensus))
        print(f"✅ Updated consensus: {consensus}/4 models required")
        
        if consensus == 0:
            print("   ⚠️  WARNING: Consensus filter disabled!")
        elif consensus == 1:
            print("   ⚠️  Very permissive (any 1 model can trigger)")
        elif consensus == 2:
            print("   ⚠️  Permissive (2/4 models = 50% agreement)")
        elif consensus == 3:
            print("   ✅ Balanced (3/4 models = 75% agreement)")
        elif consensus == 4:
            print("   🔒 Conservative (all models must agree)")
        
        print()
        print("⚠️  Bot must be restarted for changes to take effect")
        return True
    except ValueError:
        print(f"❌ Invalid consensus: {new_consensus}")
        return False

def cmd_force_entry(reason="Manual override"):
    """Create force-entry override file"""
    save_override('force_entry', True, reason)
    print("✅ Force entry enabled for next scan")
    print(f"   Reason: {reason}")
    print()
    print("⚠️  Bot will enter on next scan regardless of thresholds")
    print("   Override will auto-clear after one use")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/bot_control.py status")
        print("  python scripts/bot_control.py set-threshold 0.60")
        print("  python scripts/bot_control.py set-consensus 2")
        print('  python scripts/bot_control.py force-entry "reason here"')
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        cmd_status()
    elif command == 'set-threshold':
        if len(sys.argv) < 3:
            print("❌ Missing threshold value")
            sys.exit(1)
        cmd_set_threshold(sys.argv[2])
    elif command == 'set-consensus':
        if len(sys.argv) < 3:
            print("❌ Missing consensus value")
            sys.exit(1)
        cmd_set_consensus(sys.argv[2])
    elif command == 'force-entry':
        reason = sys.argv[2] if len(sys.argv) > 2 else "Manual override"
        cmd_force_entry(reason)
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
