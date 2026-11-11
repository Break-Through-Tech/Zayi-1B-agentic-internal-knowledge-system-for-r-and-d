"""Launch the Gradio app in the browser (November, Task 6).

Run from the repo root:
    python scripts/app.py              # http://127.0.0.1:7860
    python scripts/app.py --port 8000

Answers need GOOGLE_API_KEY in .env (see README); "Search only" works
without one. The first launch builds the index if it is missing (~1 min).
"""

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from src.ui import CSS, FAVICON, HEAD, Backend, build_app, make_theme


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    backend = Backend()
    print("Loading the knowledge base and models…", flush=True)
    backend.warm_up()
    build_app(backend).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=make_theme(),
        css=CSS,
        head=HEAD,
        favicon_path=FAVICON,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
