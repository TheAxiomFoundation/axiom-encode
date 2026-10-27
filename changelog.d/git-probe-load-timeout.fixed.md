Give RuleSpec checkout identity and configuration Git probes a configurable
10-second timeout and one retry on timeout, preventing transient host load from
rejecting otherwise valid context roots while preserving checkout validation.
