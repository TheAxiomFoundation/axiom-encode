Raise the corpus release object byte cap from 16 MiB to 64 MiB in the
resolver and the registry materializer, and apply the same cap to every
release-object fetch: the four workflow curl steps and `axiom-encode ci`.
Wave-4 union objects are 25.2 MiB pretty-printed.
