#!/bin/bash
# Quick status check for parallel training runs

echo "════════════════════════════════════════════════════════════"
echo "  QRT PARALLEL TRAINING STATUS"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if sessions exist
if tmux has-session -t single_loss 2>/dev/null; then
    echo "✅ Single Loss Session: RUNNING"
else
    echo "❌ Single Loss Session: NOT FOUND"
fi

if tmux has-session -t combined_loss 2>/dev/null; then
    echo "✅ Combined Loss Session: RUNNING"
else
    echo "❌ Combined Loss Session: NOT FOUND"
fi

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Recent Output from Single Loss:"
echo "────────────────────────────────────────────────────────────"
if tmux has-session -t single_loss 2>/dev/null; then
    tmux capture-pane -t single_loss -p | tail -10
else
    echo "  (Session not running)"
fi

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Recent Output from Combined Loss:"
echo "────────────────────────────────────────────────────────────"
if tmux has-session -t combined_loss 2>/dev/null; then
    tmux capture-pane -t combined_loss -p | tail -10
else
    echo "  (Session not running)"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Output Files Generated:"
echo "════════════════════════════════════════════════════════════"
ls -lh qrt_comparison_*.png 2>/dev/null || echo "  No plots generated yet"

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Commands:"
echo "────────────────────────────────────────────────────────────"
echo "  Monitor single loss:    tmux attach -t single_loss"
echo "  Monitor combined loss:  tmux attach -t combined_loss"
echo "  Kill all sessions:      tmux kill-server"
echo "  Run this script:        ./check_status.sh"
echo "════════════════════════════════════════════════════════════"



