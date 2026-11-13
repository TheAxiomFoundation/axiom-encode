Run the signing-supervisor checks for pull requests that target dependency
branches as well as main, so stacked supervisor changes receive those checks
before their dependencies merge. The encoder test and lint workflow still runs
only for pull requests into main; run it on a stacked branch with
`gh workflow run CI --ref <branch>`.
