"""Atomic admission and durable FIFO. Redis is the source of job state.

Only the oldest admitted job is dispatched. A pending download therefore cannot
be overtaken by later messages. Slots include downloads, waiting and running work.
"""
from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from uuid import uuid4

import redis

from .config import settings

TERMINAL = {"SUCCESS", "FAILURE", "REVOKED", "EXPIRED"}

# Every operation (including sweeping expired reservations) is one transaction.
SCRIPT = r'''
local prefix, op = KEYS[1], ARGV[1]
local data = cjson.decode(ARGV[2])
local now, cfg = tonumber(ARGV[3]), cjson.decode(ARGV[4])
local active = prefix .. 'active'
local function load(id)
  local raw = redis.call('GET', prefix .. 'job:' .. id)
  if raw then return cjson.decode(raw) end
end
local function save(j)
  local ttl = math.ceil(math.max(cfg.history, (j.deadline or now) - now + cfg.history))
  redis.call('SET', prefix .. 'job:' .. j.id, cjson.encode(j), 'EX', ttl)
end
local function finish(j, state, status)
  j.state, j.status, j.finished_at = state, status, now
  j.deadline = now
  save(j)
  redis.call('ZREM', active, j.id)
end
for _, id in ipairs(redis.call('ZRANGEBYSCORE', active, '-inf', now)) do
  local j = load(id)
  if not j then
    redis.call('ZREM', active, id)
  elseif j.state == 'DISPATCHED' and j.queue_deadline > now then
    -- A publish/worker crash must not strand the head of the queue.
    j.state, j.deadline = 'QUEUED', j.queue_deadline
    save(j)
    redis.call('ZADD', active, j.deadline, j.id)
  else
    local state = 'EXPIRED'
    if j.state == 'RUNNING' or j.state == 'CANCEL_REQUESTED' then state = 'FAILURE' end
    finish(j, state, 'Истёк срок загрузки, ожидания или обработки')
  end
end
local function jobs()
  local result = {}
  for _, id in ipairs(redis.call('ZRANGE', active, 0, -1)) do
    local j = load(id)
    if j then table.insert(result, j) end
  end
  table.sort(result, function(a, b) return a.seq < b.seq end)
  return result
end
if op == 'list' then
  local result = {}
  for _, j in ipairs(jobs()) do if j.actor == data.actor then table.insert(result, j) end end
  return cjson.encode({jobs=result})
end
if op == 'reserve' then
  if data.dedup ~= '' then
    local old = redis.call('GET', prefix .. 'dedup:' .. data.dedup)
    local existing = old and load(old)
    if existing then return cjson.encode({job=existing, duplicate=true}) end
  end
  local count = 0
  local all = jobs()
  for _, j in ipairs(all) do if j.actor == data.actor then count = count + 1 end end
  if count >= cfg.user_max then return cjson.encode({error='Лимит незавершённых задач пользователя', code=429, retry_after=5}) end
  if #all >= cfg.global_max then return cjson.encode({error='Очередь заполнена', code=503, retry_after=5}) end
  local ratekey = prefix .. 'rate:' .. data.actor
  local last = tonumber(redis.call('GET', ratekey) or '-1000000000')
  if cfg.interval > 0 and now - last < cfg.interval then
    return cjson.encode({error='Слишком частые загрузки', code=429, retry_after=math.ceil(cfg.interval - now + last)})
  end
  local j = {id=data.id, actor=data.actor, lane=data.lane, prompt=data.prompt,
    kind=data.kind, seq=redis.call('INCR', prefix .. 'sequence'), state='RESERVED',
    created_at=now, deadline=now+cfg.upload_timeout, queue_deadline=now+cfg.queue_timeout,
    status='Ожидаю загрузку файла'}
  save(j)
  redis.call('ZADD', active, j.deadline, j.id)
  redis.call('SET', ratekey, tostring(now), 'EX', math.max(1, math.ceil(cfg.interval)))
  if data.dedup ~= '' then
    redis.call('SET', prefix .. 'dedup:' .. data.dedup, j.id, 'EX', math.ceil(cfg.history + cfg.queue_timeout + cfg.process_timeout))
  end
  return cjson.encode({job=j, duplicate=false})
end
if op == 'dispatch' then
  local all = jobs()
  if #all == 0 then return '{}' end
  for _, j in ipairs(all) do
    if j.state == 'DISPATCHED' or j.state == 'RUNNING' or j.state == 'CANCEL_REQUESTED' then return '{}' end
  end
  local j = all[1]
  if j.state ~= 'QUEUED' then return '{}' end
  j.state, j.deadline = 'DISPATCHED', math.min(now+30, j.queue_deadline)
  save(j)
  redis.call('ZADD', active, j.deadline, j.id)
  return cjson.encode(j)
end
local j = load(data.id)
if not j then return cjson.encode({error='Задача не найдена', code=404}) end
local terminal = j.state == 'SUCCESS' or j.state == 'FAILURE' or j.state == 'REVOKED' or j.state == 'EXPIRED'
if op == 'get' then
  j.queue_position = 0
  if not terminal and j.state ~= 'RUNNING' and j.state ~= 'CANCEL_REQUESTED' then
    for _, other in ipairs(jobs()) do if other.seq <= j.seq then j.queue_position = j.queue_position+1 end end
  end
elseif op == 'claim_upload' then
  if j.state == 'RESERVED' then
    j.state, j.status = 'UPLOADING', 'Загрузка файла'
    save(j)
  else return cjson.encode({error='Загрузка уже принята или задача завершена', code=409, job=j}) end
elseif op == 'ready' then
  if j.state ~= 'UPLOADING' then return cjson.encode({error='Загрузка больше не принимается', code=409}) end
  j.state, j.status, j.deadline = 'QUEUED', 'В очереди', j.queue_deadline
  j.payload, j.prompt = data.payload, data.prompt
  save(j)
  redis.call('ZADD', active, j.deadline, j.id)
elseif op == 'unpublish' then
  if j.state == 'DISPATCHED' then
    j.state, j.deadline = 'QUEUED', j.queue_deadline
    save(j)
    redis.call('ZADD', active, j.deadline, j.id)
  end
elseif op == 'begin' then
  if j.state ~= 'DISPATCHED' then return '{}' end
  j.state, j.status, j.started_at = 'RUNNING', 'Обработка началась', now
  j.owner, j.deadline = data.owner, now + cfg.process_timeout + 30
  save(j)
  redis.call('ZADD', active, j.deadline, j.id)
elseif op == 'progress' then
  if j.owner == data.owner and j.state == 'RUNNING' then j.status = data.status; save(j) end
elseif op == 'cancel' then
  if j.state == 'RUNNING' or j.state == 'UPLOADING' then
    j.state, j.status = 'CANCEL_REQUESTED', 'Останавливаю обработку'
    save(j)
  elseif not terminal and j.state ~= 'CANCEL_REQUESTED' then finish(j, 'REVOKED', 'Задача отменена до обработки') end
elseif op == 'finish' then
  if not terminal and (not j.owner or j.owner == data.owner) then
    j.result = data.result
    finish(j, data.state, data.status)
  end
end
return cjson.encode(j)
'''


class QueueError(RuntimeError):
    def __init__(self, message, code=503, retry_after=5):
        super().__init__(message)
        self.code, self.retry_after = code, retry_after


class JobStore:
    def __init__(self, client, *, prefix="voiceflow:jobs:", clock=time.time):
        self.client, self.prefix, self.clock = client, prefix, clock

    def _call(self, operation, **data):
        import json
        cfg = dict(history=settings.JOB_HISTORY_SEC, global_max=settings.QUEUE_MAX_JOBS,
                   user_max=settings.QUEUE_MAX_USER_JOBS, interval=settings.QUEUE_ADMISSION_INTERVAL_SEC,
                   upload_timeout=settings.UPLOAD_TIMEOUT_SEC, queue_timeout=settings.QUEUE_TIMEOUT_SEC,
                   process_timeout=settings.PROCESS_TIMEOUT_SEC)
        try:
            raw = self.client.eval(SCRIPT, 1, self.prefix, operation, json.dumps(data), self.clock(), json.dumps(cfg))
            result = json.loads(raw)
        except redis.RedisError as exc:
            raise QueueError("Очередь временно недоступна") from exc
        if "error" in result:
            raise QueueError(result["error"], result["code"], result.get("retry_after", 5))
        return result

    def reserve(self, actor, lane, prompt, kind="audio", message_key=""):
        dedup = hashlib.sha256(f"{actor}:{message_key}".encode()).hexdigest() if message_key else ""
        return self._call("reserve", id=uuid4().hex, actor=actor, lane=lane,
                          prompt=prompt, kind=kind, dedup=dedup)

    def get(self, task_id):
        return self._call("get", id=task_id)

    def for_actor(self, actor):
        return self._call("list", actor=actor)["jobs"] or []

    def claim_upload(self, task_id):
        return self._call("claim_upload", id=task_id)

    def ready(self, task_id, payload, prompt):
        return self._call("ready", id=task_id, payload=payload, prompt=prompt)

    def dispatch(self):
        return self._call("dispatch")

    def unpublish(self, task_id):
        return self._call("unpublish", id=task_id)

    def begin(self, task_id, owner):
        return self._call("begin", id=task_id, owner=owner)

    def progress(self, task_id, owner, status):
        return self._call("progress", id=task_id, owner=owner, status=status)

    def cancel(self, task_id):
        return self._call("cancel", id=task_id)

    def finish(self, task_id, state, status, result=None, owner=""):
        return self._call("finish", id=task_id, state=state, status=status, result=result, owner=owner)


@lru_cache(maxsize=1)
def get_store():
    return JobStore(redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2))


def dispatch_ready():
    """Recoverable outbox: only one task is sent to Celery at a time."""
    from .worker import process_audio, process_transcript_only
    store = get_store()
    job = store.dispatch()
    if not job:
        return
    task = process_audio if job["kind"] == "audio" else process_transcript_only
    try:
        task.apply_async(args=[job["payload"], job["prompt"]], task_id=job["id"], retry=False)
    except Exception:
        store.unpublish(job["id"])
        raise


def public_status(job):
    state = job["state"]
    if state in {"RESERVED", "UPLOADING", "QUEUED", "DISPATCHED"}:
        state = "PENDING"
    elif state == "RUNNING":
        state = "PROGRESS"
    result = {"state": state, "status": job["status"], "task_id": job["id"],
              "created_at": job["created_at"], "started_at": job.get("started_at"),
              "queue_position": job.get("queue_position", 0)}
    if job["state"] in {"QUEUED", "DISPATCHED"} and result["queue_position"]:
        result["status"] = f"В очереди, позиция {result['queue_position']}"
    if job["state"] == "SUCCESS":
        result.update(job.get("result") or {})
    return result
