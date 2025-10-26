# LLM Extraction Module (Public Sample)

This repository contains a **sanitized, standalone version** of the LLM extraction component
developed as part of a larger confidential AI project at **White Stork**.

The full project — an end-to-end AI pipeline for legal document processing — is **proprietary** and
not publicly available. This file (`llm_extract.py`) illustrates a **modular and generic approach**
for orchestrating **Claude (Anthropic)** models via **AWS Bedrock** to extract structured JSON
data from OCR text.

### Features
- Bedrock streaming integration with Claude 3 models  
- Robust JSON extraction from streamed model outputs  
- Schema-based prompt generation and normalization  
- Modular design for different legal document types (decrees, decisions, etc.)

### Note
All sensitive or organization-specific logic has been removed to comply with confidentiality
requirements.
