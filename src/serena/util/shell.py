import os
import subprocess

from pydantic import BaseModel

from solidlsp.util.subprocess_util import subprocess_kwargs


class ShellCommandResult(BaseModel):
    stdout: str
    return_code: int
    cwd: str
    stderr: str | None = None


def subprocess_check_output(args: list[str], encoding: str = "utf-8", strip: bool = True, timeout: float | None = None) -> str:
    output = subprocess.check_output(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, env=os.environ.copy(), **subprocess_kwargs()).decode(encoding)  # type: ignore
    if strip:
        output = output.strip()
    return output
