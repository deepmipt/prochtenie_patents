## Description

The rule-based solver for selecting roles in historical texts. It extracts possible role occurrences and store them in the field  `instance["annotations"]["history_role_skill"]["roles"]` as a dict  of the form

```{"type": "РОЛЬ", "span_start": ..., "span_end": ..., "person": ..., "text": ....}```.
