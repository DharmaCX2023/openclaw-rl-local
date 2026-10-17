#!/usr/bin/env python3
"""Search the web through the configured Serper-compatible gateway."""

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 0.25
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class SerperRequestError(RuntimeError):
    pass


def _normalize_search_url(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise SerperRequestError("invalid endpoint URL")
    try:
        parsed = urllib.parse.urlsplit(value)
        _ = parsed.port
    except ValueError:
        raise SerperRequestError("invalid endpoint URL") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise SerperRequestError("invalid endpoint URL")
    normalized = value.rstrip("/")
    if not normalized.endswith("/search"):
        normalized += "/search"
    return normalized


def _request_json(request: urllib.request.Request) -> object:
    for attempt in range(MAX_ATTEMPTS):
        retryable = False
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            error_message = f"HTTP {exc.code}"
            retryable = exc.code in RETRYABLE_HTTP_CODES
        except (urllib.error.URLError, TimeoutError) as exc:
            error_message = type(exc).__name__
            retryable = True
        except json.JSONDecodeError:
            raise SerperRequestError("invalid JSON response") from None
        except (http.client.HTTPException, ValueError):
            error_message = "invalid request configuration"
            retryable = False

        if not retryable or attempt + 1 == MAX_ATTEMPTS:
            raise SerperRequestError(error_message) from None
        time.sleep(RETRY_DELAY_SECONDS * (2**attempt))

    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--gl")
    parser.add_argument("--hl")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    return args


def main() -> int:
    args = parse_args()
    search_url = os.environ.get(
        "SERPER_URL", os.environ.get("SERPER_BASE_URL", "")
    ).strip()
    api_key = os.environ.get("SERPER_API_KEY", "").strip()
    if not api_key:
        print("Serper credentials are not configured", file=sys.stderr)
        return 2
    try:
        search_url = _normalize_search_url(search_url)
    except SerperRequestError as exc:
        print(f"Serper request failed: {exc}", file=sys.stderr)
        return 2

    payload = {"q": args.query, "num": args.limit}
    if args.gl:
        payload["gl"] = args.gl
    if args.hl:
        payload["hl"] = args.hl

    request = urllib.request.Request(
        search_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        raw = _request_json(request)
    except SerperRequestError as exc:
        print(f"Serper request failed: {exc}", file=sys.stderr)
        return 1

    organic = raw.get("organic", []) if isinstance(raw, dict) else []
    results = []
    for item in organic[: args.limit]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
                "source": "web",
            }
        )

    print(
        json.dumps(
            {"query": args.query, "count": len(results), "results": results},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
