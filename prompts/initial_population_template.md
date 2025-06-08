You are an expert AI game player for Flappy Bird. Design {n} diverse starting policies.

{policy_format_guidance}

Avoid policies similar to these recent failures:
{failed_json_str}

Respond with ONLY a raw JSON array containing {n} policy objects.
The overall structure MUST be a JSON array: json```[ policy_object_1, policy_object_2, ..., policy_object_N ]```.

Example of the EXPECTED ARRAY STRUCTURE containing two policies:
json```
[
  {{
    "default": "flap",
    "dist:close_velo:stable": "do_nothing",
    "pos:above": "flap"
  }},
  {{
    "default": "do_nothing",
    "dist:far_pos:above_velo:stable": "flap",
    "velo:rising": "do_nothing"
  }}
]
```

A single policy object within the array should look like this (using the example you provided):

json```
{example_policy_json_str}
```

Make sure each policy in the array is complete and correctly formatted.
