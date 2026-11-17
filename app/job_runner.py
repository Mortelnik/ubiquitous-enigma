"""Child entry point. Heavy imports and GPU allocations stay in this process."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .queueing import get_store


class JobContext:
    def __init__(self, task_id, owner):
        self.request = SimpleNamespace(id=task_id)
        self.owner = owner

    def update_state(self, *, state, meta):
        get_store().progress(self.request.id, self.owner, meta.get("status", "Обработка"))


def main():
    from .worker import _process_audio, _process_transcript_only
    task_id, owner, result_path = sys.argv[1:]
    job = get_store().get(task_id)
    context = JobContext(task_id, owner)
    if job["kind"] == "audio":
        result = _process_audio(context, job["payload"], job["prompt"])
    else:
        text = Path(job["payload"]).read_text(encoding="utf-8")
        result = _process_transcript_only(context, text, job["prompt"], is_text=True)
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
