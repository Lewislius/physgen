# Local resource identities

These small resources remove inference's dependency on a sibling experiment's cache.

- `training_asset_inventory.json` is the exact training-time inventory whose canonical JSON digest is recorded in step0800 `complete.json`. Inference verifies the Wan source, model, VAE, T5 and tokenizer entries. Historical teacher entries are retained only so the saved inventory identity remains comparable; inference does not access teacher files.
- `negative.pt` is a byte-for-byte copy of the original cached native negative-prompt embedding. `negative.json` records its text, SHA256 and provenance. The runtime verifies the text, content hash, dimensions, dtype and finite values before using it.
- `cache_recipe.json` pins the encoding identity of historical training data. It is used only by the training/final-diagnostics cache reader; no sibling source is imported to reconstruct this identity.

No base model or checkpoint weight file was changed or duplicated here. Actual model weights remain at the paths saved in the checkpoint configuration.
