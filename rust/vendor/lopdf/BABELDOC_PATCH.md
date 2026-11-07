# BabelDOC lopdf 0.45.0 patch

Source: the offline cached crates.io lopdf 0.45.0 package (upstream license retained).

Local change: LoadOptions.real_parser accepts an optional real-token callback.
Scoped thread-local state carries it through nested dictionary parsing and each
Rayon object-stream worker, restoring prior state on return/panic. Ordinary
LoadOptions::default() retains the upstream f32 parser. BabelDOC parser_core
opts in to reproduce MuPDF 1.27.2 object-number lexing; decoded content streams
are still tokenized separately at f64 precision.

No dependency/version addition is introduced by this patch.
