# Changelog fragments

Do not edit the encoder version or `CHANGELOG.md` in a pull request. The
`Encoder version bump` workflow assigns the next version after an
encoder-affecting change merges, and compiles these fragments into
`CHANGELOG.md` under that version (axiom-encode#2048).

To record a change, add one Markdown file here, named after your branch or
pull request (for example `typed-validator-checks.md`), holding one or more
`- ` bullet lines:

```markdown
- Type validator issues from declared checks.
```

The bump removes the fragments it compiles. With no fragments, the entry lists
the subjects of the commits merged since the previous bump.
