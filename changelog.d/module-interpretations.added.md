Validate structured `module.interpretations` in encoder validation, apply, and
reviewed-candidate promotion: enforce strict shape and status-dependent evidence
fields, require non-blank kebab-case alternative IDs unique within each
interpretation, match corpus excerpts with the source-proof matcher, check local
affected rules, and block entries marked `contradicted`. Interpretation texts do
not count as proof atoms or complete-source coverage. Prompt initial encodings to
emit open entries and repairs to use only recorded rulings or publisher replies
supplied in review findings.
