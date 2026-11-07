<a name="v0.44.0"></a>
## [v0.44.0](https://github.com/J-F-Liu/lopdf/compare/v0.43.0...v0.44.0) (2026-07-10)

### Add

* Add back assets/test.pdf

### Feat

* Support incremental save of encrypted documents (#522)
* Bound text extraction against decompression bombs
* Bound load-time decompression via LoadOptions.max_decompressed_size
* Bound stream decompression to guard against bombs
* Make ttf-parser optional behind font_embedding feature

### Fix

* Reject incremental save of encrypted documents (#521)
* Fix Document::get_toc() dropping entries when outline titles repeat (#517)
* Add feature flag to font test
* Only use criterion for non-wasm tests/benches

### Move

* Move resource files in tests to assets

### Refactor

* Make `Document::get_page_content` return `Vec<u8>` since it cannot fail

### Test

* Add tests demonstrating unbounded stream decompression

### CI

* CI skip performance test
* Actually pass --no-default-features in the no-default-features matrix cell (#510)

<a name="v0.43.0"></a>
## [v0.43.0](https://github.com/J-F-Liu/lopdf/compare/v0.42.0...v0.43.0) (2026-06-25)

### Feat

* Report encryption status in PdfMetadata


<a name="v0.42.0"></a>
## [v0.42.0](https://github.com/J-F-Liu/lopdf/compare/v0.41.0...v0.42.0) (2026-06-23)

### Feat

* Add Content::decode_strict method for strict parsing

### Fix

* Limit array and dictionary nesting depth
* Include subsequent ObjStm objects in xref stream range
* Insert newline between concatenated content streams
* Consume trailing comments in content streams

### Test

* Escape trailing whitespace in parser tests


<a name="v0.41.0"></a>
## [v0.41.0](https://github.com/J-F-Liu/lopdf/compare/v0.40.0...v0.41.0) (2026-06-04)

### Add

* Add test to reproduce bug with replacing text in non-single character TJ Arrays
* Add regression test PDF file
* Add performance tests for document loading and page counting

### Fix

* Fix bug replacing text in non-single character TJ Arrays
* fix clippy errors

### Implement

* Implement differences mapping

### Release

* Release 0.41

### Removes

* Removes `nom_locate` dependency to fix a performance regression introduced in v0.35.0.


<a name="v0.40.0"></a>
## [v0.40.0](https://github.com/J-F-Liu/lopdf/compare/v0.39.0...v0.40.0) (2026-03-19)

### Fix

* Fix a few warnings.
* Fix WinAnsiEncoding handling in SimpleEncoding string conversion

### Release

* Release 0.40

### Remove

* Remove an unused dependency.

### Return

* return Option via ok() instead of unwrap() for decode_text_string

### Update

* update src/reader to make clippy happy

### Update

* Update `getrandom` to 0.4 and `rand` to 0.10.

### Upgrade

* Upgrade `wasm-bindgen-test` while we're at it.


<a name="v0.39.0"></a>
## [v0.39.0](https://github.com/J-F-Liu/lopdf/compare/v0.38.0...v0.39.0) (2026-01-16)

### Add

* add tests

### Fix

* fix conditionally compile encrypted metadata tests
* fix test_metadata and clippy warnings
* fix readme example ci issue
* fix clippy ci failed

### Release

* Release 0.39

### Breaking change


is_encrypted() now returns false after successful
decryption. Use was_encrypted() to check original encryption state.


<a name="v0.38.0"></a>
## [v0.38.0](https://github.com/J-F-Liu/lopdf/compare/v0.37.0...v0.38.0) (2026-01-16)

### Apply

* apply fix the warnings

### Inline

* Inline ImageMasks don't need to have a ColorSpace so don't require one.


<a name="v0.37.0"></a>
## [v0.37.0](https://github.com/J-F-Liu/lopdf/compare/v0.36.0...v0.37.0) (2026-01-16)

### Add

* add unicode replace support ([#421](https://github.com/J-F-Liu/lopdf/issues/421))

### Avoid

* Avoid unwrapping object streams when decrypting documents

### Change

* Change remove_object() to only remove the object ([#423](https://github.com/J-F-Liu/lopdf/issues/423))

### Ensure

* Ensure we only encrypt with compliant P-values

### Fix

* fix bug ([#422](https://github.com/J-F-Liu/lopdf/issues/422))

### Fix

* Fix clippy
* Fix bug by encoding OE, UE and Perms for revision 5

### Implement

* Implement tests to encrypt and decrypt documents
* Implement tests for the password algorithms

### Improve

* Improve readability by consistently iterating over blocks
* Improve Length field checks

### Retain

* Retain permission bits

### Set

* Set up WASI CI
* Set up WASM CI workflow

### Update

* Update changelog


<a name="v0.36.0"></a>
## [v0.36.0](https://github.com/J-F-Liu/lopdf/compare/0.35.0...v0.36.0) (2025-03-15)

### Add

* Add support for Revision 5 ([#401](https://github.com/J-F-Liu/lopdf/issues/401))
* Add more checks to the encryption/decryption logic ([#399](https://github.com/J-F-Liu/lopdf/issues/399))
* Add sanity checks for PDF encryption, add examples for decrypting/encrypting PDF files and various bug fixes ([#397](https://github.com/J-F-Liu/lopdf/issues/397))
* Add encrypt function to crypt filters
* Add support for jiff and make both chrono and time optional features

### Avoid

* Avoid parsing encrypted object streams early and correctly parse object streams upon decryption ([#385](https://github.com/J-F-Liu/lopdf/issues/385))
* Avoid decrypting cross-reference streams ([#381](https://github.com/J-F-Liu/lopdf/issues/381))

### Check

* Check if the security handler is the standard one

### Clarify

* Clarify datetime parsing logic using the PDF specification

### Compute

* Compute the file encryption key (revision 6)

### Declare

* Declare and implement crypt filters

### Ensure

* Ensure the document is actually encrypted

### Fix

* Fix unused imports
* Fix warning for rotate example
* Fix warnings about nom_parser
* Fix clippy warning about operator precedence
* Fix typo in comment

### Gracefully

* Gracefully handle the is_aes check without throwing errors ([#376](https://github.com/J-F-Liu/lopdf/issues/376))

### Handle

* Handle cases where the stream objects override the crypt filter

### Implement

* Implement Document::encrypt() ([#396](https://github.com/J-F-Liu/lopdf/issues/396))
* Implement password authentication (revision 6)
* Implement decrypt with and without password sanitization
* Implement authentication functions
* Implement password sanitization from string
* Implement password algorithms 2-7
* Implement encrypt_object function
* Implement and use PKCS[#5](https://github.com/J-F-Liu/lopdf/issues/5) padding instead
* Implement function to parse the available crypt filters
* Implement 256-bit AES-CBC crypt filter
* Implement TryFrom rather than TryInto

### Improve

* Improve the AES decryption with some sanity checks ([#383](https://github.com/J-F-Liu/lopdf/issues/383))

### Merge

* Merge remaining algorithms functions into PasswordAlgorithm implementation

### Missing

* Missing import to test

### Only

* Only encode EncryptMetadata when V >= 4 ([#400](https://github.com/J-F-Liu/lopdf/issues/400))

### Provide

* Provide revision-agnostic functions for the password algorithms

### Randomly

* Randomly generate file encryption key for V5 in encrypt example ([#403](https://github.com/J-F-Liu/lopdf/issues/403))

### Recurse

* Recurse into arrays and dictionaries to fully decrypt all strings/streams ([#378](https://github.com/J-F-Liu/lopdf/issues/378))

### Release

* Release 0.36

### Remove

* Remove the old implementation
* Remove workflow that used to enable the nom_parser feature
* Remove nom_parser feature

### Reorganize

* Reorganize encryption code

### Sanitize

* Sanitize passwords (revision 6)

### Try

* Try decrypting with an empty password

### Unpack

* Unpack objects after decrypting object streams ([#382](https://github.com/J-F-Liu/lopdf/issues/382))

### Update

* Update to nom 8.0 and nom_locate 5.0 ([#402](https://github.com/J-F-Liu/lopdf/issues/402))

### Update

* update changelog

### Use

* Use a hasher instead of allocating a Vec
* Use the new implementation to compute the file encryption key
* Use the default stream and string crypt filter if present
* Use Unix epoch if time feature is not enabled
* Use get_deref for the Kids array to handle indirect references ([#379](https://github.com/J-F-Liu/lopdf/issues/379))

### Validate

* Validate encryption dictionary for revision 5 ([#405](https://github.com/J-F-Liu/lopdf/issues/405))

### Validate

* validate binary comment during parsing ([#392](https://github.com/J-F-Liu/lopdf/issues/392))


<a name="0.35.0"></a>
## [0.35.0](https://github.com/J-F-Liu/lopdf/compare/v0.34.0...0.35.0) (2025-01-19)

### Add

* Add test for supported color types in PDF image embedding
* Add function for text chunks extraction. ([#342](https://github.com/J-F-Liu/lopdf/issues/342))

### Added

* added binary comment as attribute and for load and write. Binary Comment is gonna be important for pdf in A/2, A/3 format. ([#370](https://github.com/J-F-Liu/lopdf/issues/370))

### Allow

* Allow parsing off-spec PDF files with prefixes before the header ([#362](https://github.com/J-F-Liu/lopdf/issues/362))

### Also

* Also accept ASCII85 streams without EOD marker ([#354](https://github.com/J-F-Liu/lopdf/issues/354))

### Fix

* Fix clippy warning
* Fix incorrect image data handling in PDF content stream
* Fix BitsPerComponent calculation and improper ColorSpace mapping
* Fix incorrect color type detection for JPEG images
* Fix mulitplication overflow in ascii85 decode ([#348](https://github.com/J-F-Liu/lopdf/issues/348))
* Fix out of memory bug ([#347](https://github.com/J-F-Liu/lopdf/issues/347))
* Fix addition overflow ([#346](https://github.com/J-F-Liu/lopdf/issues/346))
* Fix lowercase s of Procset and no space target string(J-F-Liu[#323](https://github.com/J-F-Liu/lopdf/issues/323)) ([#324](https://github.com/J-F-Liu/lopdf/issues/324))

### Ignore

* Ignore space after byte index of startxref ([#371](https://github.com/J-F-Liu/lopdf/issues/371))

### Implement

* Implement ToUnicode for variadic len encodings ([#328](https://github.com/J-F-Liu/lopdf/issues/328))

### Improve

* Improve JPEG processing efficiency by avoiding unnecessary decode ([#345](https://github.com/J-F-Liu/lopdf/issues/345))
* Improve cmap parsing and internal error handling ([#335](https://github.com/J-F-Liu/lopdf/issues/335))

### Inline

* Inline images ([#356](https://github.com/J-F-Liu/lopdf/issues/356))

### Keep

* keep existing values when extending dictionary ([#322](https://github.com/J-F-Liu/lopdf/issues/322))

### Properly

* Properly support document prefixes ([#365](https://github.com/J-F-Liu/lopdf/issues/365))

### Refactor

* Refactor and optimize image processing logic in xobject.rs

### Release

* Release 0.35

### Remove

* remove misleading Object::as_string ([#350](https://github.com/J-F-Liu/lopdf/issues/350))

### Remove

* Remove superfluous `ref` keyword ([#361](https://github.com/J-F-Liu/lopdf/issues/361))
* Remove pom parser ([#355](https://github.com/J-F-Liu/lopdf/issues/355))
* Remove /Prev from trailer ([#333](https://github.com/J-F-Liu/lopdf/issues/333))

### Replace

* Replace debug assert with Result ([#349](https://github.com/J-F-Liu/lopdf/issues/349))
* Replace unwrap with error handling ([#351](https://github.com/J-F-Liu/lopdf/issues/351))

### Rework

* Rework errors ([#358](https://github.com/J-F-Liu/lopdf/issues/358))

### Specify

* Specify minimum Rust version in Cargo.toml ([#320](https://github.com/J-F-Liu/lopdf/issues/320))

### Support

* Support UTF-16 encoding for bookmark titles with non-ASCII characters ([#364](https://github.com/J-F-Liu/lopdf/issues/364))
* Support AES encryption and revision 4 ([#343](https://github.com/J-F-Liu/lopdf/issues/343))

### Throw

* Throw error if xref stream cannot be uncompressed ([#339](https://github.com/J-F-Liu/lopdf/issues/339))

### Update

* update changelog


<a name="v0.34.0"></a>
## [v0.34.0](https://github.com/J-F-Liu/lopdf/compare/v0.33.0...v0.34.0) (2024-08-31)

### Add

* Add ASCII85 decoding ([#317](https://github.com/J-F-Liu/lopdf/issues/317))
* Add text extraction based on ToUnicode cmap  ([#314](https://github.com/J-F-Liu/lopdf/issues/314))
* Add error handling to object stream ([#299](https://github.com/J-F-Liu/lopdf/issues/299))
* Add PDFDocEncoding ([#296](https://github.com/J-F-Liu/lopdf/issues/296))

### Cleanup

* Cleanup comments and cargo fmt ([#290](https://github.com/J-F-Liu/lopdf/issues/290))

### Detect

* Detect reference cycles when going through trailers ([#308](https://github.com/J-F-Liu/lopdf/issues/308))
* Detect reference cycles when parsing streams (with nom_parser) ([#300](https://github.com/J-F-Liu/lopdf/issues/300))
* Detect reference cycles when collecting page resources ([#298](https://github.com/J-F-Liu/lopdf/issues/298))

### Fix

* Fix unicode fonts extraction in extract text example. ([#315](https://github.com/J-F-Liu/lopdf/issues/315))
* Fix clippy warings

### Implement

* Implement encoding and decoding of text strings (PDF1.7 section 7.9.2.2) ([#297](https://github.com/J-F-Liu/lopdf/issues/297))

### Improve

* Improve error handling ([#307](https://github.com/J-F-Liu/lopdf/issues/307))

### Refactor

* Refactor get_or_create_resources() ([#291](https://github.com/J-F-Liu/lopdf/issues/291))

### Release

* Release 0.34

### Replace

* Replace unwrap with returning error ([#310](https://github.com/J-F-Liu/lopdf/issues/310))
* Replace LinkedHashMap with IndexMap ([#293](https://github.com/J-F-Liu/lopdf/issues/293))

### Update

* Update dependencies ([#309](https://github.com/J-F-Liu/lopdf/issues/309))
* Update readme of pdfutil ([#295](https://github.com/J-F-Liu/lopdf/issues/295))


<a name="v0.33.0"></a>
## [v0.33.0](https://github.com/J-F-Liu/lopdf/compare/v0.32.0...v0.33.0) (2024-08-31)

### Accept

* Accept comments in content parsing ([#261](https://github.com/J-F-Liu/lopdf/issues/261))

### Added

* Added a new feature to get images info from the pdf page. ([#275](https://github.com/J-F-Liu/lopdf/issues/275))

### Async

* Async Examples ([#266](https://github.com/J-F-Liu/lopdf/issues/266))

### AsyncReader

* AsyncReader ([#265](https://github.com/J-F-Liu/lopdf/issues/265))

### Fix

* Fix parse outline failed, the key ’D‘ might be an object id ([#274](https://github.com/J-F-Liu/lopdf/issues/274))
* Fix parse outline failed([#270](https://github.com/J-F-Liu/lopdf/issues/270)) ([#271](https://github.com/J-F-Liu/lopdf/issues/271))

### Indexmap

* indexmap use in TOC for sorted TOC ([#267](https://github.com/J-F-Liu/lopdf/issues/267))

### Release

* Release 0.33

### Replace

* Replace md5 with md-5 ([#272](https://github.com/J-F-Liu/lopdf/issues/272))


<a name="v0.32.0"></a>
## [v0.32.0](https://github.com/J-F-Liu/lopdf/compare/v0.31.0...v0.32.0) (2024-08-31)

### Add

* Add debug format for hexadecimal ([#240](https://github.com/J-F-Liu/lopdf/issues/240))

### Added

* Added big generation value parsing ([#257](https://github.com/J-F-Liu/lopdf/issues/257))

### Added

* added object parse to get_page_fonts ([#249](https://github.com/J-F-Liu/lopdf/issues/249))
* added meta info decryption ([#237](https://github.com/J-F-Liu/lopdf/issues/237))

### Fix

* Fix clippy warning and format code
* Fix clippy warnings
* Fix typo in README.md ([#251](https://github.com/J-F-Liu/lopdf/issues/251))

### Fixed

* Fixed parsing of the PDFs with incorrect xrefs to indirect objects ([#254](https://github.com/J-F-Liu/lopdf/issues/254))

### Fixed

* fixed clippy issues ([#238](https://github.com/J-F-Liu/lopdf/issues/238))

### Handle

* Handle references to arrays in get_page_contents() ([#245](https://github.com/J-F-Liu/lopdf/issues/245))

### Object

* Object and related types implement PartialEq ([#236](https://github.com/J-F-Liu/lopdf/issues/236))

### Release

* Release 0.32


<a name="v0.31.0"></a>
## [v0.31.0](https://github.com/J-F-Liu/lopdf/compare/v0.30.0...v0.31.0) (2023-05-10)

### Add

* Add example of page rotation ([#230](https://github.com/J-F-Liu/lopdf/issues/230))
* Add decryption of documents using RC4 encryption. ([#228](https://github.com/J-F-Liu/lopdf/issues/228))

### Annotate

* Annotate feature usage ([#229](https://github.com/J-F-Liu/lopdf/issues/229))

### Fix

* Fix typo in README.md ([#233](https://github.com/J-F-Liu/lopdf/issues/233))

### PDF

* PDF 2.0 is now a free specification

### Release

* Release 0.31

### Remove

* Remove extraneous `Q` operation from insert_image ([#227](https://github.com/J-F-Liu/lopdf/issues/227))


<a name="v0.30.0"></a>
## [v0.30.0](https://github.com/J-F-Liu/lopdf/compare/v0.29.0...v0.30.0) (2023-04-09)

### Add

* Add support for extracting TOC, Outlines and NamedDestinations ([#211](https://github.com/J-F-Liu/lopdf/issues/211))
* Add example extract_text ([#212](https://github.com/J-F-Liu/lopdf/issues/212))
* Add get_encrypted and is_encrypted ([#210](https://github.com/J-F-Liu/lopdf/issues/210))
* Add load_filtered method ([#198](https://github.com/J-F-Liu/lopdf/issues/198))
* Add as_string method to Object ([#196](https://github.com/J-F-Liu/lopdf/issues/196))

### Adding

* Adding Comments to examples ([#220](https://github.com/J-F-Liu/lopdf/issues/220))

### Fix

* Fix clippy warning
* Fix cliippy warnings
* Fix datetime using time crate
* Fix Cargo.toml ([#213](https://github.com/J-F-Liu/lopdf/issues/213))
* Fix ci build issue ([#209](https://github.com/J-F-Liu/lopdf/issues/209))
* Fix extract_text to split text at word boundaries.
* Fix embed_image feature

### Make

* Make some more objects public. ([#199](https://github.com/J-F-Liu/lopdf/issues/199))

### Readd

* Readd accidently deleted pdf files in assets ([#204](https://github.com/J-F-Liu/lopdf/issues/204))

### Release

* Release 0.30

### Remove

* Remove obsolete lifetime

### Replace

* Replace unmaitained encoding crate with encoding_rs ([#222](https://github.com/J-F-Liu/lopdf/issues/222))

### Set

* Set default to nom_parser and rayon ([#208](https://github.com/J-F-Liu/lopdf/issues/208))

### Update

* Update time dependency ([#206](https://github.com/J-F-Liu/lopdf/issues/206))
* Update nom dependency
* Update time dependency
* Update edition and some dependencies.


<a name="v0.29.0"></a>
## [v0.29.0](https://github.com/J-F-Liu/lopdf/compare/v0.27.0...v0.29.0) (2023-04-09)

### Add

* Add function get_page_annotations and include an example ([#184](https://github.com/J-F-Liu/lopdf/issues/184))

### Added

* Added documentation and improved tests ([#178](https://github.com/J-F-Liu/lopdf/issues/178))

### Allow

* Allow mutable access to the document catalog ([#189](https://github.com/J-F-Liu/lopdf/issues/189))

### Extend

* Extend match layout change and Full bookmark example in merge. ([#179](https://github.com/J-F-Liu/lopdf/issues/179))

### Fix

* Fix nom parser
* Fix clippy warnings
* Fix add_barcode example
* Fix Incremental.pdf
* Fix documentation issues and make README testable ([#171](https://github.com/J-F-Liu/lopdf/issues/171))
* Fix pdfutil build error
* Fix `extend` definition confusion bug ([#161](https://github.com/J-F-Liu/lopdf/issues/161))

### Fixed

* Fixed [#175](https://github.com/J-F-Liu/lopdf/issues/175) and some clippy issues.  ([#182](https://github.com/J-F-Liu/lopdf/issues/182))

### Guard

* Guard example based on if the "parser" feature is enabled ([#173](https://github.com/J-F-Liu/lopdf/issues/173))

### Made

* made XREF parser accept an optional space character after 'xref' ([#167](https://github.com/J-F-Liu/lopdf/issues/167))

### Make

* Make add_xobject follow references ([#187](https://github.com/J-F-Liu/lopdf/issues/187))
* Make xref public ,fix line endings and Fix Xref output so Adobe will open them again. ([#181](https://github.com/J-F-Liu/lopdf/issues/181))

### Merge

* Merge branch 'master' of https://github.com/J-F-Liu/lopdf

### Release

* Release 0.29
* Release 0.28

### Remove

* Remove --no-default-features test

### Remove

* remove unneccessary time 0.1 dependency ([#163](https://github.com/J-F-Liu/lopdf/issues/163))

### Reorder

* Reorder Pages before Renumbering Objects. ([#193](https://github.com/J-F-Liu/lopdf/issues/193))

### Support

* Support Incremental Updates ([#176](https://github.com/J-F-Liu/lopdf/issues/176))

### Switch

* switch to single-precision floating point ([#190](https://github.com/J-F-Liu/lopdf/issues/190))

### Update

* Update itoa dependency to 1.0 ([#162](https://github.com/J-F-Liu/lopdf/issues/162))


<a name="v0.27.0"></a>
## [v0.27.0](https://github.com/J-F-Liu/lopdf/compare/v0.26.0...v0.27.0) (2021-12-16)

### Add

* Add GitHub Actions build matrix ([#127](https://github.com/J-F-Liu/lopdf/issues/127))
* Add Change Log

### Added

* Added Object::as_float() to convert numerical values to float. ([#124](https://github.com/J-F-Liu/lopdf/issues/124))
* Added Object::as_bool ([#123](https://github.com/J-F-Liu/lopdf/issues/123))

### Avoid

* Avoid panic when encounters negative stream length

### Bookmarks

* Bookmarks ([#135](https://github.com/J-F-Liu/lopdf/issues/135))

### Change

* Change indent_style to space

### Check

* Check stream length

### Do

* Do not limit Real precision to two digits ([#155](https://github.com/J-F-Liu/lopdf/issues/155))

### Fix

* Fix document save race in parser_aux::load_and_save and creator::create_document ([#151](https://github.com/J-F-Liu/lopdf/issues/151))
* Fix clippy warnings & add clippy build job ([#128](https://github.com/J-F-Liu/lopdf/issues/128))

### Preserve

* Preserve the eol characters in literal strings ([#131](https://github.com/J-F-Liu/lopdf/issues/131))

### Reduce

* Reduce allocation by reusing the iterator ([#129](https://github.com/J-F-Liu/lopdf/issues/129))

### Release

* Release 0.27
* Release pdfutil 0.4

### Replace

* Replace lzw with weezl ([#140](https://github.com/J-F-Liu/lopdf/issues/140))

### Return

* Return early on error in `Stream::filters` ([#130](https://github.com/J-F-Liu/lopdf/issues/130))

### Unwrap

* Unwrap the text ([#119](https://github.com/J-F-Liu/lopdf/issues/119))

### Update

* Update nom to 6.0 ([#126](https://github.com/J-F-Liu/lopdf/issues/126))


<a name="v0.26.0"></a>
## [v0.26.0](https://github.com/J-F-Liu/lopdf/compare/v0.25.0...v0.26.0) (2020-09-29)

### Add

* Add as_str, as_str_mut methods to Object ([#107](https://github.com/J-F-Liu/lopdf/issues/107))

### Dtoa

* dtoa may write real number in exponential format which is not allowed in PDF

### Genericize

* Genericize Content to allow AsRef<[Operation]> ([#111](https://github.com/J-F-Liu/lopdf/issues/111))

### Make

* Make pom dependency optional (but default) ([#112](https://github.com/J-F-Liu/lopdf/issues/112))
* Make rayon dependency optional ([#108](https://github.com/J-F-Liu/lopdf/issues/108))

### Merge

* Merge document PDF logic with some fixes ([#117](https://github.com/J-F-Liu/lopdf/issues/117))

### Various

* Various improvements, updated libraries and image features ([#118](https://github.com/J-F-Liu/lopdf/issues/118))


<a name="v0.25.0"></a>
## [v0.25.0](https://github.com/J-F-Liu/lopdf/compare/v0.24.0...v0.25.0) (2020-06-25)

### Add

* add indexing checks ([#98](https://github.com/J-F-Liu/lopdf/issues/98))

### Add

* Add a test for [#93](https://github.com/J-F-Liu/lopdf/issues/93) ([#95](https://github.com/J-F-Liu/lopdf/issues/95))

### Bugfix

* Bugfix for xref_start. ([#105](https://github.com/J-F-Liu/lopdf/issues/105))

### Check

* check  that the buffer is big enough for startxref ([#93](https://github.com/J-F-Liu/lopdf/issues/93))

### Create

* Create rust.yml ([#104](https://github.com/J-F-Liu/lopdf/issues/104))

### Extend

* extend recursion limit to non-local references ([#100](https://github.com/J-F-Liu/lopdf/issues/100))

### Fix

* Fix compilation error&test error ([#102](https://github.com/J-F-Liu/lopdf/issues/102))

### Keep

* keep looking for the last pattern ([#94](https://github.com/J-F-Liu/lopdf/issues/94))

### Limit

* Limit allowed bracket depth. ([#97](https://github.com/J-F-Liu/lopdf/issues/97))

### Limit

* limit recursion to the number of objects ([#92](https://github.com/J-F-Liu/lopdf/issues/92))

### Move

* Move bracket depth checking into parsers. ([#101](https://github.com/J-F-Liu/lopdf/issues/101))

### Release

* Release 0.25

### Return

* Return Result from as_array_mut() ([#106](https://github.com/J-F-Liu/lopdf/issues/106))

### Update

* Update itoa and linked-hash-map ([#91](https://github.com/J-F-Liu/lopdf/issues/91))


<a name="v0.24.0"></a>
## [v0.24.0](https://github.com/J-F-Liu/lopdf/compare/v0.23.0...v0.24.0) (2020-02-17)

### Compute

* Compute an accurate iterator size when the page tree is sane.

### Fix

* Fix datetime parser ([#89](https://github.com/J-F-Liu/lopdf/issues/89))

### More

* More permissive datetime parsing ([#90](https://github.com/J-F-Liu/lopdf/issues/90))

### Release

* Release 0.24

### Validate

* Validate expected id in pom parser.
* Validate the expected id when reading indirect objects.


<a name="v0.23.0"></a>
## [v0.23.0](https://github.com/J-F-Liu/lopdf/compare/v0.22.0...v0.23.0) (2019-07-14)

### Adapt

* Adapt pom parser.

### Add

* Add error descriptions.
* Add a proper error type and remove some more panics.

### Allow

* Allow loading a document from a memory slice.

### Avoid

* Avoid allocating an intermediate collection for iteration.
* Avoid unwraps when already returning an Option for failure.

### Error

* Error signaling around compression and image handling.

### Escape

* Escape fix ([#68](https://github.com/J-F-Liu/lopdf/issues/68))

### Export

* Export dereference function as it is useful for PDF consumers.
* Export filters module.

### Get_font_encoding

* get_font_encoding seems more at home with Dictionary.

### Handle

* Handle stream filter chains ([#66](https://github.com/J-F-Liu/lopdf/issues/66))

### Hex

* Hex fix ([#67](https://github.com/J-F-Liu/lopdf/issues/67))

### Implement

* Implement LZW decompression.

### Improve

* Improve hex parsing performance.

### Make

* Make a page iterator.
* Make Reader::read consume the Reader.
* Make content operations faillible.

### Protect

* Protect against reference loops.
* Protect against a corrupted page tree.

### Refactor

* Refactor a bit to allow a utility function.

### Release

* Release 0.23.0

### Remove

* Remove intermediate assignation.
* Remove unsafe code around FilterType.
* Remove unsafe code on get_object_mut.
* Remove some 'if let' for readability.
* Remove more panic paths in xref parsing.

### Replace

* Replace unwraps in processor.rs.

### Return

* Return results when appropriate.

### Separate

* Separate decompression into two functions.

### Take

* Take care of panic that I actually hit on the pom side.
* Take care of creator.rs.

### Unify

* Unify buffer creation.

### Use

* Use lifetime ellision.
* Use TryInto.
* Use writeln where appropriate.
* Use error enum in reader.
* Use stable cloned.


<a name="v0.22.0"></a>
## [v0.22.0](https://github.com/J-F-Liu/lopdf/compare/v0.21.0...v0.22.0) (2019-05-13)

### Add

* Add parsing benchmark.
* Add nom dependency.

### Also

* Also test with nom parsing feature enabled.

### Array

* Array and dictionary parsing.

### Avoid

* Avoid using format! when writing.

### Be

* Be explicit about trait objects.

### Boolean

* Boolean and null parsing.

### Content

* Content parsing.

### Duplicate

* Duplicate pom parser for incremental replacement with nom 5.

### Ease

* Ease off on rayon a bit.

### Escape

* Escape sequence parsing.

### Extern

* extern crate is not required anymore with 2018 edition.

### Fix

* Fix last ugly parser.
* Fix octal parser.
* Fix pdfutil build

### Float

* Float parsing.

### Header

* Header parsing.

### Hex

* Hex string parsing.

### Indirect

* Indirect object and stream parsing.

### Literal

* Literal string syntax.

### Make

* Make sure Stream.start_position is relative to the whole file.

### Merge

* Merge remote-tracking branch 'upstream/master' into nom5
* Merge remote-tracking branch 'upstream/master' into nom5

### More

* More 2018 edition lints.
* More cleanup.
* More cleanup.
* More simplifications.

### Object

* Object id and reference parsing.

### Octal

* Octal and hexadecimal parsing.

### Parallel

* Parallel object stream parsing.

### Rayon

* Rayon usage proof of concept.

### Release

* Release 0.22.0

### Remove

* Remove pom dependency in tests.

### Replace

* Replace name parser.

### Resolve

* Resolve name collisions.

### Simplify

* Simplify lifetime annotations.

### Slowly

* Slowly replace the cute pom parser with nom.

### Trailer

* Trailer and xref start.

### Turns

* Turns out "contained" already exists in nom.

### Unify

* Unify both variants of the parsing functions.

### Use

* Use a BufWriter when saving to path.
* Use parse_at(&self.buffer, offset) to read indirect_object
* Use nom digit testing functions.
* Use lifetime ellision.
* Use nom sequence operators.

### Useless

* Useless move.

### Xref

* Xref stream and trailer parsing.
* Xref parsing.


<a name="v0.21.0"></a>
## [v0.21.0](https://github.com/J-F-Liu/lopdf/compare/v0.20.0...v0.21.0) (2019-04-26)

### Avoid

* Avoid allocating a String.

### Check

* Check offsets read from file to avoid panics
* Check and correct Size entry of trailer dictionary

### Clean

* Clean up bytes_to_string, string_to_bytes iterators

### Fix

* Fix clippy warnings
* Fix .editorconfig

### Fixed

* fixed finally

### Redundant

* Redundant imports with 2018 edition.

### Release

* Release 0.21.0

### Update

* Update example
* Update Cargo.toml

### Use

* Use env_logger in pdfutil


<a name="v0.20.0"></a>
## [v0.20.0](https://github.com/J-F-Liu/lopdf/compare/v0.19.0...v0.20.0) (2019-03-07)

### Release

* Release 0.20.0

### Replace

* Replace println with log macros

### Use

* Use Rust 2018
* Use pom 3.0


<a name="v0.19.0"></a>
## [v0.19.0](https://github.com/J-F-Liu/lopdf/compare/v0.18.0...v0.19.0) (2018-10-24)

### Allow

* Allow xref section has zero entries

### Dictionary

* Dictionary key type changed to Vec<u8>

### Format

* Format code with rustfmt

### Improve

* Improve codestyle (simplify loops, remove closures, use is_empty() etc.)

### Move

* Move image dependency to embed_image feature

### Release

* Release 0.19.0

### Skip

* Skip corrupt deflate stream


<a name="v0.18.0"></a>
## [v0.18.0](https://github.com/J-F-Liu/lopdf/compare/v0.17.0...v0.18.0) (2018-10-05)

### Able

* Able to read stream when it's length is in object stream

### Adress

* Adress timezone formatting problem from [#34](https://github.com/J-F-Liu/lopdf/issues/34)

### Insert

* insert image on page


<a name="v0.17.0"></a>
## [v0.17.0](https://github.com/J-F-Liu/lopdf/compare/v0.16.0...v0.17.0) (2018-09-19)

### Make

* Make chrono crate optional

### Release

* Release 0.17.0

### Update

* Update add_barcode example


<a name="v0.16.0"></a>
## [v0.16.0](https://github.com/J-F-Liu/lopdf/compare/v0.15.3...v0.16.0) (2018-09-18)

### Add

* Add form xobject to page
* Add extract_stream subcommand

### Compress

* Compress created Form xobject

### Compress

* compress page content after change

### Fix

* Fix collect_fonts_from_resources for referenced resources
* Fix add xobject to page resources as direct object


<a name="v0.15.3"></a>
## [v0.15.3](https://github.com/J-F-Liu/lopdf/compare/v0.15.0...v0.15.3) (2018-09-14)

### Decompress

* Decompress Form XObject

### Disable

* Disable auto format markdown

### Fix

* Fix bug in reading incremental updated document
* Fix build warning
* Fix string_to_bytes method

### Hexadecimal

* Hexadecimal strings can contain white space.

### Remove

* Remove println in extract_text

### Update

* Update example code
* Update example


<a name="v0.15.0"></a>
## [v0.15.0](https://github.com/J-F-Liu/lopdf/compare/v0.14.1...v0.15.0) (2018-02-04)

### Add

* add `get_object_mut`
* add method as_array_mut

### Extract

* Extract text from specified pages

### Replace

* Replace text of specified page


<a name="v0.14.1"></a>
## [v0.14.1](https://github.com/J-F-Liu/lopdf/compare/v0.13.0...v0.14.1) (2017-11-03)

### Add

* Add `impl From<_> for Object` for more numeric types
* Add an Object::string_literal constructor
* Add a `dictionary!` macro that creates a Dictionary
* Add `impl From<ObjectId> for Object` creating Object::Reference

### Derive

* Derive Clone for lopdf::Document

### Release

* Release 0.14.0

### Remove

* Remove the Seek bound on Document::save_to


<a name="v0.13.0"></a>
## [v0.13.0](https://github.com/J-F-Liu/lopdf/compare/v0.11.0...v0.13.0) (2017-10-02)

### Avoid

* Avoid decompress flate stream which has Subtype

### Debug

* Debug with lldb

### Fix

* Fix get_object for created document

### Ignore

* Ignore invalid objects when reading all object in xref table

### Impl

* impl fmt::Debug for Object

### Pdfutil

* pdfutil add extract_pages command

### Read

* Read optional space at the end of xref subsection header line

### Release

* Release 0.13.0
* Release 0.12.0

### Store

* Store compressed stream objects and normal objects together


<a name="v0.11.0"></a>
## [v0.11.0](https://github.com/J-F-Liu/lopdf/compare/v0.10.0...v0.11.0) (2017-08-21)

### Release

* Release 0.11.0

### Use

* Use itoa and dtoa to improve writing performance


<a name="v0.10.0"></a>
## [v0.10.0](https://github.com/J-F-Liu/lopdf/compare/v0.9.0...v0.10.0) (2017-07-20)

### Added

* Added optional allows_compression for Stream object

### Release

* Release 0.10.0


<a name="v0.9.0"></a>
## [v0.9.0](https://github.com/J-F-Liu/lopdf/compare/v0.8.0...v0.9.0) (2017-05-24)

### Add

* Add pdfutil readme

### Added

* Added unit test for load_from() and save_to()
* Added Document::with_version + refactored save() and load()
* Added Debug trait for lopdf::Document

### Apply

* Apply multiple operations in one command

### Build

* Build with Rust stable
* Build with Rust beta

### Fix

* Fix delete_zero_length_streams

### Fixed

* Fixed unit tests
* Fixed breaking API changes

### Release

* Release 0.9.0


<a name="v0.8.0"></a>
## [v0.8.0](https://github.com/J-F-Liu/lopdf/compare/v0.7.0...v0.8.0) (2017-03-16)

### Change

* Change Name(String) to Name(Vec<u8>)

### Delete_object

* delete_object and delete_unused_objects

### Get_pages

* get_pages and delete_pages

### Handle

* Handle zero length stream

### Release

* Release 0.8.0

### Traverse

* Traverse objects from trailer recursively


<a name="v0.7.0"></a>
## [v0.7.0](https://github.com/J-F-Liu/lopdf/compare/v0.6.0...v0.7.0) (2017-03-07)

### Add

* Add Content::decode() function

### Build

* Build on Rust 1.17

### Create

* Create String object for DateTime

### Parse

* Parse PDF datetime value

### Read

* Read xref stream in hybrid-reference file

### Update

* Update create_document example
* Update README


<a name="v0.6.0"></a>
## [v0.6.0](https://github.com/J-F-Liu/lopdf/compare/v0.5.0...v0.6.0) (2017-02-16)

### Add

* Add Stream::decompressed_content() method

### Read

* Read previous Xrefs of linearized or incremental updated document


<a name="v0.5.0"></a>
## [v0.5.0](https://github.com/J-F-Liu/lopdf/compare/v0.4.0...v0.5.0) (2017-02-10)

### Add

* Add size field to Xref
* Add Xref struct

### Decode

* Decode PNG frame after FlateDecode

### Read

* Read compressed objects from object stream
* Read xref stream

### Update

* Update README

### Use

* Use pom 0.9.0

### XrefEntry

* XrefEntry as enum type


<a name="v0.4.0"></a>
## [v0.4.0](https://github.com/J-F-Liu/lopdf/compare/v0.3.0...v0.4.0) (2017-01-29)

### Add

* Add Operation constructor
* Add modify_text test
* Add travis-ci build status
* Add FAQ in Readme
* Add print_xref_size() for debuging

### Decode

* Decode content stream

### Encode

* Encode content operations

### Fix

* Fix load_document test
* Fix https://github.com/rust-lang/rust/issues/39177

### Optimize

* Optimize parser code

### Solve

* Solve mutual reference problem between Pages and Page objects

### Trigger

* Trigger new release to pass build on docs.rs

### Update

* Update create PDF example


<a name="v0.3.0"></a>
## [v0.3.0](https://github.com/J-F-Liu/lopdf/compare/v0.2.0...v0.3.0) (2017-01-18)

### Add

* Add compress/decompress subcommands to pdfutil

### Create

* create PDF parser using pom instead of nom

### Dictionary

* Dictionary preserve key insert order

### Update

* Update README
* Update parser to use pom 0.6.0

### Use

* Use reader to get stream length if it is a reference object


<a name="v0.2.0"></a>
## [v0.2.0](https://github.com/J-F-Liu/lopdf/compare/v0.1.0...v0.2.0) (2017-01-05)

### Add

* Add pdfutil program

### Fix

* Fix parsing PDF array error

### Improve

* Improve documentation


<a name="v0.1.0"></a>
## v0.1.0 (2016-12-23)

### Editor

* Editor config

### Impl

* impl Document add_object method

### Improve

* Improve Document::save functional type

### Initial

* Initial commit

### PDF

* PDF objects and document definition

### Parse

* Parse and load PDF document

### Read

* Read objects from xref table instead of sequentially from file stream

### Save

* Save PDF document to file

### Store

* Store max_id as a field of document

