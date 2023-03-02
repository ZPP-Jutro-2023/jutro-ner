import os
from pathlib import Path

import typer
from doccano_client import DoccanoClient
from dotenv import load_dotenv


def download(project_id: int, dir_name: Path, url: str):
    load_dotenv()


    # instantiate a client and log in to a Doccano instance
    client = DoccanoClient(url)
    client.login(username=os.getenv("DOCCANO_USERNAME"), password=os.getenv("DOCCANO_PASSWORD"))

    file_path = client.download(project_id=project_id, format="JSONL", only_approved=True, dir_name=dir_name)
    file_path.rename('assets/dataset.zip')

if __name__ == "__main__":
    typer.run(download)