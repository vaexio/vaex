import base64
import binascii
import sys
import requests
import time
from requests.api import head
from vaex.file import exists
import vaex.utils
import contextlib
import hashlib
import os
import vaex.settings

# rich = vaex.utils.optional_import('rich', modules=[rich.progress])


def make_progress():
    import rich
    import rich.progress

    progress = rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.TimeElapsedColumn(),
        rich.progress.FileSizeColumn(),
        rich.progress.TransferSpeedColumn(),
    )
    return progress


def download(url, sql_query=None, state=None, format="arrow", f=None, directory=None, verbose=True, check=True, debug=False, client=None):
    if sql_query is not None:
        params = {"query": sql_query, "output": format}
    elif state is not None:
        import json

        state_encoded = json.dumps(state)
        params = {"state": state_encoded, "output": format}
    path = None
    client = client or requests
    if f is None:
        response = client.head(url, params=params)
        if debug:
            print("HTTP headers (HEAD)", response.headers)
        if "ETag" not in response.headers:
            raise ValueError(f"{url} did not reply with ETag")
        filename = response.headers["ETag"]
        filename = f"{filename}.{format}"
        if directory:
            path = os.path.join(directory, filename)
        else:
            path = filename
        if verbose:
            print("Downloading to", path)
        file_context = open(path, "ab+")
        response.close()
    else:
        if vaex.file.is_path_like(f):
            if verbose:
                print("Downloading to", f)
                path = f
            file_context = vaex.file.open(f, "ab+")
        elif vaex.file.is_file_object(f):
            file_context = contextlib.nullcontext(f)
        else:
            raise TypeError("f argument is not a path or file")

    with file_context as f:
        offset = 0
        headers = {}
        start = offset = f.tell()
        if check:
            hasher = hashlib.sha256()
            f.seek(0)
            while True:
                data = f.read(1024 ** 2)
                hasher.update(data)
                if not data:
                    break
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if debug:
            print("Sending HTTP get headers", headers)
        response = client.get(url, params=params, headers=headers, stream=True)
        response.raise_for_status()
        t0 = time.time()
        if debug:
            print("HTTP headers", response.headers)
        if verbose:
            progress = make_progress()
            if path:
                description = f"Downloading: {path}"
            else:
                description = f"Downloading"
            # 1024TB is assumed the max, we don't know the final size
            task_id = progress.add_task(description, total=int(1024 ** 5))
            progress.start()
        try:
            for chunk in response.iter_content(chunk_size=1024 ** 2):
                offset += len(chunk)
                if check:
                    hasher.update(chunk)

                t1 = time.time()
                dt = t1 - t0
                # print(f"\r{offset//1024**3}GB {(offset-start)/1024**2/dt:.2f}MB/s", end="")
                if verbose:
                    progress.update(task_id, completed=(offset - start))
                f.write(chunk)
                f.flush()
        finally:
            if verbose:
                progress.stop()
    if check:
        response = client.head(url, params=params, headers={"Want-Digest": "sha-256"})
        if debug:
            print("HTTP headers (HEAD)", response.headers)
        if verbose:
            print("Validating file...", end="")
        hash_local = hasher.hexdigest()
        _algo, encoded = response.headers["Digest"].split("=", 1)
        # server does base64 encoding
        hash_server = base64.decodebytes(encoded.encode("ascii"))
        hash_server = binascii.hexlify(hash_server).decode("ascii")
        if hash_server != hash_local:
            raise ValueError(f"Hash is not equal {hash_server} (server) != {hash_local} (download)")
        if verbose:
            print("\rsha256 hash", hash_local)
    return path


if __name__ == "__main__":
    directory = os.path.join(vaex.settings.data.path, "remote")
    os.makedirs(directory, exist_ok=True)
    path = download(sys.argv[1], sys.argv[2], directory=directory, verbose=True)
    print(vaex.open(path))