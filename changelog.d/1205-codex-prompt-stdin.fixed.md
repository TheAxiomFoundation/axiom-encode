Send the Codex prompt on stdin instead of argv in the encode backend and the prompt-eval runner, so a large source text no longer fails process start with E2BIG (#1205).
