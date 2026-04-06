# backup_fp16_io8_inline_ln1p

This branch is a backup/demo profile.
- Keep Top as the only shared SRAM owner
- Keep runtime-configurable model structure
- Switch external data path to io8 serialized profile
- Switch SRAM packing toward 8x16-bit word layout
- Prioritize local runnable demo over mainline closure
