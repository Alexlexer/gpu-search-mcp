# Inspect scope before indexing

```sh
python -m gpu_service.index_scope --directory /path/to/repo --max-entries 500
```

This metadata-only command does not load PyTorch/models, read source contents,
write caches or start a watcher. Each displayed file has an inclusion/exclusion
reason; excluded directories are pruned and represented by one entry. Output
contains local paths and sizes, so treat it as local project metadata.

The exact-search scanner uses the same discovery policy. Defaults: supported
extensions, 5 MiB maximum per file, `.env*` disabled, internal caches, build and
research artifact directories excluded. Directory links/junctions and file links
outside the root are not indexed. No silent partial traversal on directory errors.
The report is a snapshot; it is not a security guarantee against concurrent edits.

`--max-entries` bounds report records (1–100000). When truncated, counts/bytes are
explicitly **shown-entry totals**, not estimated totals for the repository. OS
directory enumeration itself can still use memory proportional to a directory's
entry count. `--allow-env-files` previews explicit environment-file inclusion;
it never prints their contents. Inspector flags do not persist server settings.

The current policy does **not** honor `.gitignore` or custom glob rules yet; the
report explicitly states `gitignore_honored: false`. Other index engines can have
different extensions/policies. Do not use this as a claim that every engine has
the identical scope. Configurable shared ignore policies are follow-up work.
