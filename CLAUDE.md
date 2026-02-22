# Claude Development Guidelines

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update "tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to tasks/todo.md
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Project-Specific Patterns

### Plotly Chart Label Positioning
When adding labels/annotations to payoff diagrams or other charts, use paper coordinates to avoid overlap:

**Pattern:**
- **Above chart area**: `y=1.08, yref="paper", yanchor="bottom"` (e.g., underlying price)
- **Inside chart area**: Use default positioning with `annotation_position="top"` (e.g., breakeven labels)
- **Below chart area**: `y=-0.08, yref="paper", yanchor="top"` (e.g., midpoint price)

**Example:**
```python
# Label above chart
fig.add_annotation(
    x=price,
    y=1.08,  # Above chart area
    yref="paper",
    text=f"${price:,.2f}",
    showarrow=False,
    font=dict(size=9, color='#00bfff', weight='bold'),
    yanchor="bottom"
)

# Label below chart
fig.add_annotation(
    x=price,
    y=-0.08,  # Below chart area
    yref="paper",
    text=f"${price:,.2f}",
    showarrow=False,
    font=dict(size=9, color='#ff4444', weight='bold'),
    yanchor="top"
)
```

**Margin Requirements:**
- Increase top margin when adding labels above: `margin=dict(t=35)` (default: 25)
- Increase bottom margin when adding labels below: `margin=dict(b=40)` (default: 30)

### Order Fill Price Extraction (Schwab API)
When extracting fill prices from Schwab order data:

**Key Points:**
- `orderActivityCollection.executionLegs` contains actual fill prices per leg
- `legId` field is **1-indexed** (not 0-indexed!)
- Map `legId` to `orderLegCollection` using: `leg_index = legId - 1`
- For multi-fill orders, use first activity's prices for consistency
- Order-level `price` field contains net credit/debit for spreads

**Example:**
```python
activities = order.get('orderActivityCollection', [])
if activities:
    activity = activities[0]  # Use first fill
    for exec_leg in activity.get('executionLegs', []):
        leg_id = exec_leg.get('legId', 0)  # 1-indexed!
        price = exec_leg.get('price', 0)

        if leg_id > 0:
            leg_index = leg_id - 1  # Convert to 0-indexed
            leg = order_legs[leg_index]
            symbol = leg.get('instrument', {}).get('symbol', '')
            fill_prices[symbol] = price
```

### Iron Condor Midpoint Calculation
For iron condors and 4-leg strategies:
- Midpoint = average of all short strikes (negative quantity legs)
- Display as red dotted vertical line with label below x-axis
- Helps visualize if underlying is still centered in profit zone

**Example:**
```python
short_strikes = [
    leg.get('strike')
    for leg in legs
    if leg.get('quantity', 0) < 0 and leg.get('strike')
]
if len(short_strikes) >= 2:
    midpoint = sum(short_strikes) / len(short_strikes)
```
