#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║   fix_csi.sh — CSI Camera Recovery Script for QCar2         ║
# ║   Cleans up stale camera resources and restarts nvargus      ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage:   sudo bash fix_csi.sh
#          (requires sudo for nvargus-daemon restart)
#
# Run this whenever CSI cameras stop responding, hang, or produce
# blank frames — typically after a script crash or unclean Ctrl+C.

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   QCar2 CSI Camera Recovery                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# ── Step 1: Kill zombie Python processes holding camera resources ──
echo "[1/4] Killing zombie camera processes..."
KILLED=0
for PID in $(pgrep -f "QCarCameras|Camera2D|camera_bridge|test_csi|nvargus" -u nvidia 2>/dev/null); do
    CMD=$(ps -p "$PID" -o args= 2>/dev/null | head -c 80)
    echo "  Killing PID $PID: $CMD"
    kill -9 "$PID" 2>/dev/null && KILLED=$((KILLED + 1))
done

if [ "$KILLED" -eq 0 ]; then
    echo "  No zombie processes found ✅"
else
    echo "  Killed $KILLED process(es)"
fi
echo

# ── Step 2: Restart nvargus-daemon ──
echo "[2/4] Restarting nvargus-daemon (clears stale NVMM sessions)..."
if command -v systemctl &>/dev/null; then
    systemctl restart nvargus-daemon
    echo "  nvargus-daemon restarted ✅"
else
    echo "  ⚠️  systemctl not available, trying manual restart..."
    killall -9 nvargus-daemon 2>/dev/null || true
    sleep 1
    /usr/sbin/nvargus-daemon &
    echo "  nvargus-daemon started manually"
fi
echo

# ── Step 3: Wait for daemon stabilization ──
echo "[3/4] Waiting 3 seconds for daemon to stabilize..."
sleep 3
echo "  Done ✅"
echo

# ── Step 4: Final Instructions ──
echo "[4/4] Hardware initialized globally."
echo
echo "═══════════════════════════════════════════════════════"
echo "  🟢 CSI CAMERA RECOVERY COMPLETE"
echo "  The nvargus-daemon is reset and ready."
echo "  You can now run your test scripts normally:"
echo "    python3 test_csi_all.py"
echo "═══════════════════════════════════════════════════════"
