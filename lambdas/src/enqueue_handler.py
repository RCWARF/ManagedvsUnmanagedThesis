"""S3 ObjectCreated -> SQS FIFO. One message per object; message group per object."""
import hashlib
import json
from urllib.parse import unquote_plus

from rag.clients import sqs
from rag.metrics import emit
from rag.settings import SETTINGS
from saaf.Inspector import Inspector


def lambda_handler(event, context):
    inspector = Inspector()
    inspector.inspectContainer()  # newcontainer flag: cold starts per stage
    records = event.get("Records", [])
    inspector.addAttribute("num_s3_records", len(records))

    entries, sent, failed = [], 0, 0

    def flush():
        nonlocal entries, sent, failed
        if not entries:
            return
        resp = sqs.send_message_batch(QueueUrl=SETTINGS.queue_url, Entries=entries)
        sent += len(resp.get("Successful", []))
        failed += len(resp.get("Failed", []))
        entries = []

    for i, rec in enumerate(records):
        obj = rec["s3"]["object"]
        msg = {
            "bucket": rec["s3"]["bucket"]["name"],
            "key": unquote_plus(obj["key"]),
            "etag": obj.get("eTag"),
            "size": obj.get("size"),
            "sequencer": obj.get("sequencer"),
            "eventTime": rec.get("eventTime"),
        }
        body = json.dumps(msg, sort_keys=True)
        entries.append({
            "Id": str(i),
            "MessageBody": body,
            # Group per object: one object is ordered, different objects run in parallel.
            "MessageGroupId": hashlib.sha256(f"{msg['bucket']}:{msg['key']}".encode()).hexdigest()[:128],
            # Body includes eventTime + sequencer, so a re-upload is a new message.
            "MessageDeduplicationId": hashlib.sha256(body.encode()).hexdigest(),
        })
        if len(entries) == 10:
            flush()
    flush()

    inspector.addAttribute("messages_sent", sent)
    inspector.addAttribute("messages_failed", failed)
    out = emit(inspector, SETTINGS, "enqueue")
    if failed:
        # Fail the invocation so S3's async retry re-delivers the event.
        raise RuntimeError(f"{failed} SQS sends failed")
    return out
