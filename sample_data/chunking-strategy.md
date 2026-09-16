# Chunking strategy

A passage has to be small enough to fit next to several other passages in a
prompt, and large enough to still make sense read on its own. This service
uses a recursive character splitter to balance the two.

## How the split works

The splitter tries a list of separators in order of preference: a blank
line (paragraph break), a single newline, then sentence-ending punctuation,
then a plain space, and only as a last resort a hard cut in the middle of a
word. It walks the text trying the first separator; wherever a piece is
still longer than the target chunk size, it re-splits that piece with the
next separator down the list. The result is that a chunk boundary almost
always falls on a paragraph or sentence edge, because those separators are
tried before anything cruder.

## Size and overlap

The default target is 900 characters per chunk with a 150-character overlap
between consecutive chunks. The overlap exists so a sentence that would
otherwise land right at a chunk boundary still appears in full inside at
least one chunk — without it, a fact split across the boundary could be
unretrievable no matter how the query is phrased.

## What this trades off

900 characters is roughly two to three short paragraphs — enough for a
retrieved passage to carry real context, but small enough that four
retrieved passages still fit comfortably inside a language model's prompt
alongside the question and instructions. A corpus of long, prose-heavy
documents might want a larger chunk size; a corpus of short, dense
reference entries (a glossary, a table of specs) might want a smaller one
and less overlap, since there's less risk of a fact spanning two chunks.
