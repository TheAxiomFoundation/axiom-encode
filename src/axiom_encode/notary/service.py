"""Bounded ASGI adapters. Select exactly one service role per deployment.

TLS terminates at the custodian's proxy; bind uvicorn to a private Unix socket.
Never enable HTTP access/debug logs containing Authorization or request bodies.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ._schema import decode_base64, fields
from .bundle import pack_bundle
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .identity import IdentityRefusal
from .protocol import parse_artifact

MAX_REQUEST = 131072


def create_app(
    *, signer_factory=None, broker=None, read_gateway=None, approval_loader=None
):
    if sum(value is not None for value in (signer_factory, broker, read_gateway)) != 1:
        raise ValueError("one capability role per service process")
    operations = (
        {"receipt", "transition", "genesis"}
        if signer_factory
        else {"read"}
        if read_gateway
        else {"publish-tokens", "release", "finalize", "read"}
    )

    async def invoke(request: Request):
        operation = request.path_params["operation"]
        headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        try:
            if (
                operation not in operations
                or request.query_params
                or request.headers.get("content-type", "").split(";")[0]
                != "application/json"
            ):
                raise IdentityRefusal("service_request")
            if len(request.headers.getlist("authorization")) != 1:
                raise IdentityRefusal("service_authorization")
            auth = request.headers["authorization"]
            if not auth.startswith("Bearer ") or not 0 < len(auth[7:]) <= 32768:
                raise IdentityRefusal("service_authorization")
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > MAX_REQUEST:
                    raise IdentityRefusal("service_request_size")
            body = strict_parse(bytes(raw))
            if not isinstance(body, dict):
                raise IdentityRefusal("service_request")

            def execute():
                if read_gateway:
                    return read_gateway.read(auth[7:], body)
                if signer_factory:
                    if not fields(body, {"candidate_sha256", "approval_base64"}):
                        raise IdentityRefusal("service_sign_request")
                    signer = signer_factory()  # Fresh protected-tip policy.
                    if body["approval_base64"] is None and approval_loader is not None:
                        # The approver job authenticates before discovering
                        # inbox state. No signing occurs until a hardware
                        # sidecar over this exact completed digest exists.
                        signer._authenticate(auth[7:], body["candidate_sha256"])
                        approval = approval_loader(body["candidate_sha256"])
                        if approval is None:
                            return {"state": "awaiting_hardware_approval"}
                    else:
                        approval = decode_base64(body["approval_base64"])
                    if approval is None or len(approval) > 8192:
                        raise IdentityRefusal("service_approval")
                    bundle = getattr(signer, operation)(
                        auth[7:], body["candidate_sha256"], approval
                    )
                    matches = [
                        raw
                        for name, raw in bundle.items()
                        if name.endswith(".json")
                        and parse_artifact(raw, operation) is not None
                    ]
                    if len(matches) != 1:
                        raise IdentityRefusal("service_signed_bundle")
                    return strict_parse(
                        pack_bundle(operation, sha256_hex(matches[0]), bundle)
                    )
                if operation == "release":
                    if not fields(body, {"lease_owner"}):
                        raise IdentityRefusal("service_release_request")
                    return broker.release(auth[7:], body["lease_owner"])
                if operation == "read":
                    return broker.read(auth[7:], body)
                if body != {}:
                    raise IdentityRefusal("service_request")
                return (
                    broker.publish_tokens(auth[7:])
                    if operation == "publish-tokens"
                    else broker.finalize(auth[7:])
                )

            result = await run_in_threadpool(execute)
            return Response(
                jcs_dumps(result), media_type="application/json", headers=headers
            )
        except IdentityRefusal as exc:
            # Only controlled refusal identifiers, never API/HTTP/key material.
            import re

            reason = str(exc)
            if re.fullmatch(r"[a-z_]{1,100}", reason) is None:
                reason = "request_refused"
            return Response(
                jcs_dumps({"error": reason}),
                status_code=403,
                media_type="application/json",
                headers=headers,
            )
        except Exception:
            return Response(
                b'{"error":"service_unavailable"}',
                status_code=503,
                media_type="application/json",
                headers=headers,
            )

    return Starlette(
        debug=False, routes=[Route("/v1/{operation}", invoke, methods=["POST"])]
    )
