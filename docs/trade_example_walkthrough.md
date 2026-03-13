# Trade Example Walkthrough — Target Validation

Minute-by-minute walkthrough of a single IC trade showing how TP10/TP25/TP50 targets are populated using the race logic (each TP level races independently against SL).

## Trade Setup

| Field | Value |
|-------|-------|
| Strategy | IC_05d_25w (5-delta short, $25 wings) |
| Entry time | 2023-02-27 14:06 ET |
| SPX at entry | 3998.37 |
| Short Call | 4025 |
| Short Put | 3970 |
| Long Call | 4050 |
| Long Put | 3945 |
| Credit (mid) | $0.5500 |

## Exit Thresholds

| Threshold | Formula | Value |
|-----------|---------|-------|
| SL | 2x credit | $1.1000 |
| TP10 | credit * 0.90 | $0.4950 |
| TP25 | credit * 0.75 | $0.4125 |
| TP50 | credit * 0.50 | $0.2750 |

**Logic**: Walk forward from entry+1 through 3:00 PM. Each bar, check SL first (debit >= SL threshold), then TP (debit <= TP threshold). First to trigger wins the race. Each TP level is a separate, independent race against SL.

If neither TP nor SL triggers by 3:00 PM, the trade is **carried to close**:
- Final debit < credit → `close_win` (target=1)
- Final debit >= credit → `close_loss` (target=0)

This "carry to close" matters because many trades slowly decay toward zero but never hit the aggressive TP50 target. Rather than calling those a loss, we recognize that holding to expiration is a valid exit — you still collected the credit. The model needs to see these as wins to learn the real probability distribution.

## Minute-by-Minute Price Action

```
Time     SC_mid   SP_mid   LC_mid   LP_mid   IC_debit  Event
-------- -------- -------- -------- -------- --------- ----------------
14:06     0.3250   0.3750   0.0750   0.0750    0.5500  ← ENTRY (credit)
14:07     0.3500   0.3250   0.0750   0.0750    0.5250
14:08     0.3250   0.3750   0.0750   0.0750    0.5500
14:09     0.3000   0.3500   0.0750   0.0750    0.5000
14:10     0.2750   0.3500   0.0500   0.0750    0.5000
14:11     0.2750   0.3500   0.0500   0.0750    0.5000
14:12     0.3750   0.3000   0.0750   0.0750    0.5250
14:13     0.3250   0.3250   0.0250   0.0750    0.5500
14:14     0.3250   0.3250   0.0500   0.0750    0.5250
14:15     0.2750   0.3500   0.0750   0.0750    0.4750  ← TP10 HIT (0.475 <= 0.495)
14:16     0.2250   0.3750   0.0750   0.0750    0.4500
14:17     0.1750   0.3750   0.0250   0.0750    0.4500
14:18     0.1750   0.5500   0.0250   0.1250    0.5750  ← SP spikes, debit jumps
14:19     0.1750   0.5250   0.0250   0.1250    0.5500
14:20     0.1500   0.6000   0.0250   0.1250    0.6000
14:21     0.1250   0.6000   0.0250   0.1250    0.5750
14:22     0.1250   0.5750   0.0250   0.1250    0.5500
14:23     0.1250   0.6500   0.0250   0.1000    0.6500
14:24     0.1250   0.6000   0.0250   0.1000    0.6000
14:25     0.1250   0.6000   0.0250   0.0750    0.6250
14:26     0.1250   0.6250   0.0250   0.1000    0.6250
14:27     0.1250   0.4500   0.0250   0.0750    0.4750  ← SP drops back
14:28     0.1250   0.4250   0.0250   0.0750    0.4500
14:29     0.1250   0.4500   0.0250   0.0750    0.4750
14:30     0.1250   0.4500   0.0250   0.0750    0.4750
14:31     0.1250   0.4250   0.0250   0.0750    0.4500
14:32     0.1250   0.4250   0.0250   0.0750    0.4500
14:33     0.1250   0.4250   0.0250   0.0750    0.4500
14:34     0.1250   0.4250   0.0250   0.0750    0.4500
14:35     0.1250   0.4250   0.0250   0.0750    0.4500
14:36     0.1250   0.3500   0.0250   0.0750    0.3750  ← TP25 HIT (0.375 <= 0.4125)
14:37     0.1250   0.3750   0.0250   0.0750    0.4000
14:38     0.1250   0.4500   0.0250   0.0750    0.4750
14:39     0.1250   0.3750   0.0250   0.0750    0.4000
14:40     0.1250   0.4250   0.0250   0.0750    0.4500
14:41     0.1250   0.4500   0.0250   0.0750    0.4750
14:42     0.1000   0.4750   0.0250   0.0750    0.4750
14:43     0.0750   0.7000   0.0250   0.1250    0.6250  ← SP spikes again
14:44     0.0750   0.8500   0.0250   0.1250    0.7750
14:45     0.0750   0.7250   0.0250   0.1000    0.6750
14:46     0.0750   1.0500   0.0250   0.1250    0.9750  ← approaching SL
14:47     0.0750   1.2000   0.0250   0.1250    1.1250  ← TP50→SL HIT (1.125 >= 1.10)
14:48     0.0750   0.8500   0.0250   0.1000    0.8000  ← would have recovered...
14:49     0.0750   0.8250   0.0250   0.1000    0.7750
14:50     0.0750   0.8000   0.0250   0.0750    0.7750
14:51     0.0750   0.8000   0.0250   0.1250    0.7250
14:52     0.0750   0.8500   0.0250   0.1250    0.7750
14:53     0.0750   0.7250   0.0250   0.0750    0.7000
14:54     0.0750   0.8000   0.0250   0.0750    0.7750
14:55     0.0750   0.6750   0.0250   0.0750    0.6500
14:56     0.0750   0.7000   0.0250   0.0750    0.6750
14:57     0.0750   0.7500   0.0250   0.0750    0.7250
14:58     0.0750   0.6500   0.0250   0.0750    0.6250
14:59     0.0750   0.6250   0.0250   0.0750    0.6000
15:00     0.0750   0.7750   0.0250   0.0750    0.7500
```

## Target Output (what gets written to target_v2.parquet)

| Field | TP10 | TP25 | TP50 |
|-------|------|------|------|
| target | 1 | 1 | 0 |
| exit_reason | tp | tp | sl |
| exit_time | 14:15 | 14:36 | 14:47 |
| exit_debit | 0.4750 | 0.3750 | 1.1250 |

## What This Shows

1. **Each TP level is a separate race against SL.** TP10 hit at 14:15, TP25 hit at 14:36, but TP50 never hit — SL got there first at 14:47.

2. **SL is checked first each bar.** At 14:47, debit was 1.125 which is >= 1.10 (SL). Even though TP50 threshold (0.275) was never reached, we check SL first, so TP50's race ends with a stop loss.

3. **The short put drove all the action.** SC decayed from 0.325 to 0.075 (good), but SP spiked from 0.375 to 1.20 at the worst point (SPX dipping toward the 3970 short put). This is typical 0DTE behavior — one side decays while the other can spike hard.

4. **Carry to close**: If neither TP nor SL had hit by 15:00, we'd check the final debit (0.75) vs credit (0.55). Since 0.75 > 0.55, that would be `close_loss` (target=0). But if final debit had been say 0.40, that would be `close_win` (target=1) — you held to expiration and the credit decayed enough to profit even without hitting the TP threshold. This matters because many trades slowly decay but never hit an aggressive target like TP50.
