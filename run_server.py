import argparse
import os
from dotenv import load_dotenv
import uvicorn

from dotenv import load_dotenv
load_dotenv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.getenv("VOICE_BRIDGE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("VOICE_BRIDGE_PORT", "8787")))
    p.add_argument("--env-file", default=os.getenv("VOICE_BRIDGE_ENV_FILE", ".env"))
    args = p.parse_args()

    if args.env_file and os.path.exists(args.env_file):
        load_dotenv(args.env_file, override=False)

    uvicorn.run("voice_bridge.server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
