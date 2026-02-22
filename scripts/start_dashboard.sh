#!/bin/bash
# Start the Command Center dashboard

cd "$(dirname "$0")/.."

echo "🚀 Starting Command Center Dashboard..."
echo ""
echo "Dashboard will be available at: http://127.0.0.1:8051"
echo ""
echo "To fix 'chunk loading' errors:"
echo "  1. Close all browser tabs with 127.0.0.1:8051"
echo "  2. Clear browser cache (Cmd+Shift+R on Mac)"
echo "  3. Or use Incognito/Private mode"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

venv/bin/python3 monitoring/dashboards/command_center.py
