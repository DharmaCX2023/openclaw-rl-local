# Method figure — revision prompts

Built-in imagegen editing; no CLI fallback. Revisions correct scientific connectivity, not experiment results.

## Revision 1

```text
Correct h → Teacher → repair connectivity and verifier PASS/FAIL exits.
```

## Revision 2 (final layout)

```text
Use case: precise-object-edit. Edit the MOST RECENT method figure. Preserve the title, panel borders, palette, all formulas, the entire rightmost panel 3, and both bottom ribbons exactly. Make one targeted correction: replace the tangled middle verification diagram with a simple, accurate two-row comparison and a decision.

Inside panel 2, keep header "2 Execute and verify". Remove the existing cylinder, braces, all arrows and diagram boxes BELOW that header and ABOVE the bottom footnote. Redraw this interior cleanly:

At top, one caption: "Two isolated clones of the same snapshot W".
Then two wide, separate rounded cards with ample vertical spacing:
Upper pale coral card: heading "Original branch"; next line "c− → Real result → Frozen Student → V−".
Lower pale teal card: heading "Repair branch"; next line "c+ → Real result → Frozen Student → V+".
Treat these two cards as a comparison table: DO NOT draw connecting arrows between the two cards, or any arrows from the cards to other objects. Internal left-to-right arrows are part of the textual sequence only.
Below both cards, centered caption "Matched continuation budgets".
Then one centered outlined decision box "Compare: V+ > V− ?".
From this decision box, draw ONLY TWO output arrows:
- diagonal down-left teal arrow labelled "YES" into "Verified pair" with "(h, c+, c−)" below it;
- diagonal down-right gray arrow labelled "NO" into "No repair supervision".
Both output arrows MUST start directly at the decision box and end at their own outcome box. No arrow starts at a V+ box or a Student box. Outcomes are side by side. Leave the bottom footnote "Reject unstable or regressive pairs".
Keep every label entirely inside panel 2 with no overlap. You may slightly decrease only the middle-panel type size if needed, but maintain readability.

One tiny additional correction in panel 1: replace the existing double-headed Hindsight dashed line by a simple short caption "With hindsight" immediately above the Frozen teacher box, with NO extra arrow. Keep state h → teacher → repair arrows unchanged.

Do not alter panel 3, no new claims, no extra results. Clear scientific typography, no clipping or crossing text.
```
