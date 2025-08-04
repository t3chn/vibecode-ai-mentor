# vibecode-ai-mentor
VibeCode AI Mentor – an agentic assistant that analyzes code, suggests improvements, and teaches best practices using TiDB Cloud and LLMs.

## Overview

VibeCode AI Mentor is an agentic assistant that helps developers improve code quality by indexing codebases, retrieving similar patterns via TiDB Cloud's vector search, and using language models to generate actionable recommendations.

## Features

- **Agentic Workflow**: Multi-step retrieval and generation pipeline built on top of TiDB Cloud and Large Language Models (LLMs).
- **Retrieval-Augmented Generation**: Combines vector search for code snippets with LLM reasoning to suggest improvements.
- **Procedural Design**: Implementation follows functional programming principles for clarity and maintainability.
- **Modular Components**: Separate functions for indexing repositories, searching for relevant snippets, and generating recommendations.

## Architecture

1. **Indexing** – Extracts code snippets from a repository, computes embeddings, and stores them in a TiDB Cloud vector index.
2. **Search** – Accepts a natural language query and retrieves the most relevant code examples using vector similarity search.
3. **Recommendation** – Feeds the retrieved snippets to an LLM to generate a concise summary of best practices and improvement suggestions.

A high-level dataflow diagram:

`Developer Code → Index Repository → Vector Index (TiDB) → Search → Retrieved Snippets → Language Model → Recommendations`

## Setup

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Create a `.env` file with your API keys and TiDB Cloud credentials.
4. Run the example script: `python src/main.py`.

## Contributing

Contributions are welcome! Please follow the project style guidelines:

- Use functional or procedural programming (no classes).
- Write clear, self-documenting code.
- Follow the principles of Tidy First by keeping modules small and focused.

## License

This project is licensed under the MIT License.
