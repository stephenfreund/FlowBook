---
description: Audit and sync FORMAL_DEVELOPMENT.md with source code in both directions
---

Perform a bidirectional audit between FORMAL_DEVELOPMENT.md and the source code:

1. Read FORMAL_DEVELOPMENT.md and extract every formal-concept-to-source-code mapping.
2. **Spec → Code check:** For each formal concept, verify the referenced source code exists and correctly implements the concept. Flag any spec entries where the code is missing, outdated, or inconsistent.
3. **Code → Spec check:** Search the codebase for implementations of formal concepts that are NOT yet listed in the mapping. Flag any untracked implementations.
4. For any discrepancies:
   - If the spec defines something the code doesn't implement, implement it or flag it as a TODO.
   - If the code implements something not in the spec, add the mapping to FORMAL_DEVELOPMENT.md.
   - If the spec and code disagree, treat the spec as the source of truth and update the code.
5. Summarize all changes made.

Use a subagent for the codebase scan to keep context clean. As you work, report to the user the exact steps you are doing.
