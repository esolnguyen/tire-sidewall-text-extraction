#!/usr/bin/env python3
"""
Tire Information Extraction System

Usage:
    python main.py [--port 8000] [--host 0.0.0.0] [--reload]
"""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Tire Information Extraction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn

    os.makedirs(os.path.join("data", "output"), exist_ok=True)
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload, reload_dirs=["."])


if __name__ == "__main__":
    main()
