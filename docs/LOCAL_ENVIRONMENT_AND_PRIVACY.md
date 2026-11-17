# Authorized local environment configuration

The public repository is independent of the private D-drive dataset and contains no original photos, prompts, checkpoint weights or user-specific absolute paths.

Before running local notebooks set these variables in the authorized execution environment:

- ENDEK_RESEARCH_ROOT — directory containing the private ENDEK_Dataset and other preprocessing input folders
- ENDEK_V2_ROOT — private V2 experiment root containing TRAINING_SPLITS_PREPROCESSED_V2
- ENDEK_METADATA_CSV — private metadata CSV with source-prompt descriptions

Do not share the values of these variables publicly. The notebooks included in this repository are sanitized **copies**; original working notebooks and data remain unchanged.

To regenerate research figures, run scripts/build_ieee_access_visual_evidence.py inside the authorized V2 project hierarchy.

Additional optional reproduction inputs required by some scripts:

- ENDEK_PREVIOUS_NOTEBOOK — private path to the legacy notebook used for comparison (not part of V2 public release)
- ENDEK_HF_CACHE — a private, writable Hugging Face model-cache directory on the authorized D drive
