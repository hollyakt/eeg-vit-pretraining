#!/bin/bash
# Monitor finetune and eval jobs; save eval results when available
# 
# Improvements:
#  - Timeout if job is stuck in PENDING > 1 hour
#  - Alert on job FAILED state
#  - Remove duplicate/invalid conditions
#  - Better logging and error reporting
#
# Usage:
#   ./monitor_eval_and_notify.sh <FINETUNE_JOB> <EVAL_JOB> [POLL_INTERVAL_SECS] [MAX_PENDING_SECS]
#

FINETUNE_JOB=${1:-493160}
EVAL_JOB=${2:-493161}
POLL_INTERVAL=${3:-60}
MAX_PENDING_SECS=${4:-3600}  # Default: 1 hour max pending time
PROJECT=$(pwd)
LOGDIR="$PROJECT/logs"
mkdir -p "$LOGDIR"

PENDING_START=$(date +%s)
PENDING_TIME_LOGGED=0

while true; do
    CURRENT_TIME=$(date +%s)
    PENDING_ELAPSED=$((CURRENT_TIME - PENDING_START))
    
    # Check eval job state via squeue first
    state=""
    if squeue -j "$EVAL_JOB" -h >/dev/null 2>&1; then
        state=$(squeue -j "$EVAL_JOB" -h -o "%T" | tr -d '\r')
    else
        # fallback to sacct
        state=$(sacct -j "$EVAL_JOB" --format=State --noheader 2>/dev/null | tail -n1 | awk '{print $1}' || true)
    fi

    # Log state with timestamp
    echo "[$(date '+%a %b %d %H:%M:%S %Z %Y')] Eval job $EVAL_JOB state: $state" >> "$LOGDIR/monitor_eval.log"

    case "$state" in
        COMPLETED)
            echo "[$(date)] ✓ Eval job completed successfully." >> "$LOGDIR/monitor_eval.log"
            # copy results if present
            if [ -f pretrain_finetune_results/eval_results.json ]; then
                cp pretrain_finetune_results/eval_results.json "$LOGDIR/eval_results_$(date +%Y%m%d_%H%M%S).json" 2>/dev/null || true
                echo "[$(date)] ✓ Eval results copied to $LOGDIR" >> "$LOGDIR/monitor_eval.log"
            else
                echo "[$(date)] ⚠ Eval results file not found yet." >> "$LOGDIR/monitor_eval.log"
            fi
            # also save last lines of eval log
            if [ -f "$LOGDIR/eval_after_finetune_${EVAL_JOB}.out" ]; then
                tail -200 "$LOGDIR/eval_after_finetune_${EVAL_JOB}.out" > "$LOGDIR/eval_summary_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true
            fi
            break
            ;;
        FAILED)
            echo "[$(date)] ✗ ERROR: Eval job FAILED!" >> "$LOGDIR/monitor_eval.log"
            if [ -f "$LOGDIR/eval_after_finetune_${EVAL_JOB}.err" ]; then
                echo "[$(date)] Error log:" >> "$LOGDIR/monitor_eval.log"
                tail -50 "$LOGDIR/eval_after_finetune_${EVAL_JOB}.err" >> "$LOGDIR/monitor_eval.log"
            fi
            break
            ;;
        CANCELLED)
            echo "[$(date)] ✗ Eval job was cancelled." >> "$LOGDIR/monitor_eval.log"
            break
            ;;
        PENDING)
            if [ $PENDING_ELAPSED -gt $MAX_PENDING_SECS ]; then
                echo "[$(date)] ✗ ERROR: Job stuck in PENDING for ${PENDING_ELAPSED}s (> ${MAX_PENDING_SECS}s)" >> "$LOGDIR/monitor_eval.log"
                echo "[$(date)] Cancelling job $EVAL_JOB..." >> "$LOGDIR/monitor_eval.log"
                scancel "$EVAL_JOB" 2>/dev/null || true
                break
            elif [ $((PENDING_ELAPSED % 600)) -eq 0 ] && [ $PENDING_TIME_LOGGED -eq 0 ]; then
                # Log pending time every 10 minutes
                echo "[$(date)] ⏳ Still pending (${PENDING_ELAPSED}s / ${MAX_PENDING_SECS}s max)" >> "$LOGDIR/monitor_eval.log"
                PENDING_TIME_LOGGED=1
            else
                PENDING_TIME_LOGGED=0
            fi
            ;;
        RUNNING)
            PENDING_START=$(date +%s)  # Reset timer once job starts running
            ;;
        *)
            if [ -z "$state" ]; then
                echo "[$(date)] ⚠ Warning: Could not determine job state (job may not exist)" >> "$LOGDIR/monitor_eval.log"
            else
                echo "[$(date)] Job state: $state" >> "$LOGDIR/monitor_eval.log"
            fi
            ;;
    esac

    sleep "$POLL_INTERVAL"
done

echo "[$(date)] Monitor script finished." >> "$LOGDIR/monitor_eval.log"

