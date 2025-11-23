# Data Preparation Omnilingual
Processing Famous Persian Datasets for the OmniLingual Dataset
## vhdm/persian-voice-v1
```bash
python hf_dataset_ingestion_example.py run_persian_voice_cv ../dataset --name persian_voice --version 0
```
for MAC users:
```bash
RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1 python hf_dataset_ingestion_example.py run_persian_voice_cv ../dataset --name persian_voice --version 0
```