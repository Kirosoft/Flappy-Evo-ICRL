CRITICAL Policy Formatting Rules:
1. Output MUST be a valid JSON structure.
2. The policy object maps state strings to action strings. Action strings MUST be EXACTLY "flap" or "do_nothing". Other variations like "do noting" are INVALID.
3. A "default" key with action 'flap' or 'do_nothing' is MANDATORY (e.g., "default": "flap").
4. State strings are composed of 1, 2, or 3 parts, joined by underscores ('_'). Each part: prefix:value.
   - Position: 'pos:' (values: 'above', 'aligned', 'below') <--- EMPHASIZE THESE ARE THE ONLY VALUES
   - Distance: 'dist:' (values: 'far', 'medium', 'close') <--- EMPHASIZE
   - Velocity: 'velo:' (values: 'rising', 'stable', 'falling') <--- EMPHASIZE
   Example 1-part: "pos:aligned"
5. For 2-part or 3-part state strings (e.g., "pos:X_dist:Y" or "dist:X_pos:Y_velo:Z"),
   the individual component strings (like "dist:X", "pos:Y", "velo:Z")
   MUST BE ALPHABETICALLY SORTED before being joined by underscores.
   - Example: Given components "pos:aligned" and "dist:medium":
     - "dist:medium" comes alphabetically before "pos:aligned".
     - So, the key MUST be "dist:medium_pos:aligned".
     - "pos:aligned_dist:medium" is INCORRECT.
   - Example: Given components "velo:rising", "pos:above", "dist:far":
     - Sorted: "dist:far", then "pos:above", then "velo:rising".
     - So, the key MUST be "dist:far_pos:above_velo:rising".
     - Any other order like "pos:above_dist:far_velo:rising" is INCORRECT.
6. More specific rules (3-part > 2-part > 1-part > default) take precedence.
7. EACH state part (e.g., "pos:X", "dist:Y", "velo:Z") must describe ONLY ONE condition.
   - DO NOT combine values within a single part.
   - INCORRECT: "velo:rising_stable" (trying to say rising AND stable in one part)
   - If you want to express conditions for 'velo:rising' AND 'velo:stable' separately,
     they must be SEPARATE keys in the policy if they are 1-part rules, or part of
     different multi-part rules. A single state string can only represent one value per category (pos, dist, velo).
8. Ensure the value matches the prefix.
   - INCORRECT: "pos:rising" (position cannot be 'rising')
   - CORRECT: "velo:rising"
   