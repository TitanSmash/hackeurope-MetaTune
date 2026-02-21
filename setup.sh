#!/bin/bash
set -e

echo "==> Setting up backend..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

echo "==> Setting up frontend..."
cd metatune-ai
npm install
cd ..

echo ""
echo "Done! To run locally:"
echo "  Terminal 1: source .venv/bin/activate && cd backend && uvicorn api:app --reload --port 8000"
echo "  Terminal 2: cd metatune-ai && npm run dev"
echo "  Then open:  http://localhost:8080"
