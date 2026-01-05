# PDFMiner Changes

This directory contains a modified version of `pdfminer.six`.

## Rationale
The `pdfminer` code has been vendored directly into `babeldoc` to support specific customizations required for the project.

## Proposed Changes
1.  **Refactor**: It is recommended to extract the changes made to `pdfminer` and maintain them as a patch set or a separate Fork.
2.  **Current State**: The code references `babeldoc.pdfminer` internally.
3.  **Action Item**: Compare with upstream `pdfminer.six` and isolate changes.

(This file was created to document the technical debt as requested).
