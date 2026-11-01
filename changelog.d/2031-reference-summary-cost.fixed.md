Skip the parse in a warmed reference summary when imports cannot change the module's reads, reuse one parse otherwise, and build the imported-name set once per module.
