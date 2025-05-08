## ImGrep – Image Content Search Engine ##

ImGrep is a content-based image search engine that allows users to upload images and retrieve visually similar results. Built with Flask, Flutter, and PostgreSQL, it uses OpenAI’s CLIP model for semantic image embedding and supports secure authentication via OAuth 2.0. The system is fully containerized using Docker for scalable deployment.

## Features
🔐 OAuth 2.0 user authentication (Google, GitHub, etc.)

📤 Image upload via a Flutter mobile frontend

🧠 Image feature extraction using OpenAI CLIP

🔎 Content-based image retrieval (CBIR)

🗂️ PostgreSQL for metadata and vector storage

📁 Profile image storage and retrieval

🐳 Dockerized backend and database for easy deployment

## Tech Stack
- Frontend: Flutter (Dart)

- Backend: Flask (Python), SQLAlchemy ORM

- ML Model: OpenAI CLIP (via PyTorch)

- Database: PostgreSQL

- Authentication: OAuth 2.0 (Authlib, oauth2_client)

- Containerization: Docker & Docker Compose

