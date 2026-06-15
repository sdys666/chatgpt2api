from __future__ import annotations

import time
import uuid
import html
import json
import re
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.chat_completion_cache import cache_key, chat_completion_cache, normalize_text_messages
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    count_message_image_tokens,
    count_message_text_tokens,
    count_text_tokens,
    encode_images,
    normalize_messages,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    text_backend,
)
from services.protocol.web_search_tool import (
    WEB_SEARCH_TOOL_TYPES,
    has_unsupported_tools,
    has_web_search_tool,
    normalized_sources,
    run_web_search,
    search_query_from_messages,
    text_with_url_citations,
)
from utils.helper import extract_image_from_message_content, extract_response_prompt, has_response_image_generation_tool
from utils.image_tokens import (
    count_image_content_tokens,
    count_image_output_items_tokens,
    image_usage,
    token_usage,
)

TOOL_UNAVAILABLE_SYSTEM_MESSAGE = (
    "This compatibility backend cannot execute local tools, shell commands, non-search tools, "
    "or file operations. Do not claim to have run tools or inspected external resources. "
    "If a user asks you to use a tool, say that tool execution is unavailable through this backend."
)

TOOL_CALL_SYSTEM_MESSAGE = """Tool output adapter: when calling tools, output ONLY this XML and no prose/markdown:
<tool_calls><tool_call><tool_name>TOOL_NAME</tool_name><parameters><PARAM><![CDATA[value]]></PARAM></parameters></tool_call></tool_calls>"""

RESPONSE_CONTENT_PART_TYPES = {"text", "input_text", "output_text", "image_url", "input_image", "image"}


def is_text_response_request(body: dict[str, Any]) -> bool:
    return not has_response_image_generation_tool(body)


def has_unsupported_response_tools(body: dict[str, Any]) -> bool:
    return has_unsupported_tools(body, {"image_generation", *WEB_SEARCH_TOOL_TYPES})


def response_client_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip()
        if tool_type == "image_generation" or tool_type in WEB_SEARCH_TOOL_TYPES:
            continue
        result.append(tool)
    return result


def _tool_meta(tool: dict[str, Any]) -> tuple[str, str, object]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(tool.get("name") or fn.get("name") or tool.get("type") or "").strip()
    desc = str(tool.get("description") or fn.get("description") or "").strip()
    schema = tool.get("parameters") or tool.get("input_schema") or fn.get("parameters") or fn.get("input_schema") or {}
    return name, desc, schema


def build_tool_prompt(tools: list[dict[str, Any]]) -> str:
    blocks = []
    for tool in tools:
        name, desc, schema = _tool_meta(tool)
        if not name:
            continue
        blocks.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(schema, ensure_ascii=False)}")
    if not blocks:
        return ""
    return "Available tools:\n" + "\n".join(blocks) + f"""

Tool use rules:
- If the user asks to inspect files, edit code, run commands, or answer from local project state, call a suitable tool first.
- {TOOL_CALL_SYSTEM_MESSAGE}
- Put parameters under <parameters> using the exact schema names.
""".strip()


def strip_tool_markup(text: str) -> str:
    return re.sub(r"(?is)<tool_calls\b[^>]*>.*?</tool_calls>|<tool_call\b[^>]*>.*?</tool_call>|<function_call\b[^>]*>.*?</function_call>|<invoke\b[^>]*>.*?</invoke>", "", text or "").strip()


def xml_value(text: str, tag: str) -> str:
    match = re.search(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", text)
    if not match:
        return ""
    value = match.group(1).strip()
    cdata = re.fullmatch(r"(?is)<!\[CDATA\[(.*?)]]>", value)
    return html.unescape(cdata.group(1) if cdata else value).strip()


def parse_tool_value(raw: str) -> object:
    value = xml_value(f"<x>{raw}</x>", "x")
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_tool_params(raw: str) -> dict[str, object]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {m.group(1): parse_tool_value(m.group(2)) for m in re.finditer(r"(?is)<([\w.-]+)\b[^>]*>(.*?)</\1>", raw)}


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, object]]]:
    text = re.sub(r"(?is)```.*?```", "", text or "").strip()
    blocks = re.findall(r"(?is)<tool_call\b[^>]*>(.*?)</tool_call>|<function_call\b[^>]*>(.*?)</function_call>|<invoke\b[^>]*>(.*?)</invoke>", text)
    result = []
    for block in (next((part for part in match if part), "") for match in blocks):
        name = xml_value(block, "tool_name") or xml_value(block, "name") or xml_value(block, "function")
        params = xml_value(block, "parameters") or xml_value(block, "input") or xml_value(block, "arguments") or "{}"
        if name:
            result.append((name, parse_tool_params(params)))
    return result


def response_image_tool(body: dict[str, Any]) -> dict[str, object]:
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "image_generation":
            return tool
    return {}


def extract_response_image(input_value: object) -> tuple[bytes, str] | None:
    if isinstance(input_value, dict):
        if str(input_value.get("type") or "").strip() == "input_image":
            images = extract_image_from_message_content([input_value])
            return images[0] if images else None
        images = extract_image_from_message_content(input_value.get("content"))
        return images[0] if images else None
    if not isinstance(input_value, list):
        return None
    for item in reversed(input_value):
        if isinstance(item, dict):
            if str(item.get("type") or "").strip() == "input_image":
                images = extract_image_from_message_content([item])
                if images:
                    return images[0]
            images = extract_image_from_message_content(item.get("content"))
            if images:
                return images[0]
    return None


def _input_image_parts(input_value: object) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if isinstance(input_value, dict):
        content = input_value.get("content")
        if isinstance(content, list):
            parts.extend(item for item in content if isinstance(item, dict))
        return parts
    if not isinstance(input_value, list):
        return parts
    if all(isinstance(item, dict) and item.get("type") for item in input_value):
        return [item for item in input_value if isinstance(item, dict)]
    for item in input_value:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                parts.extend(part for part in content if isinstance(part, dict))
    return parts


def _is_response_content_part(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    part_type = str(value.get("type") or "").strip()
    return part_type in RESPONSE_CONTENT_PART_TYPES or ("image_url" in value and part_type != "message")


def _message_content_from_response_item(item: dict[str, Any]) -> object:
    item_type = str(item.get("type") or "").strip()
    if item_type == "function_call_output":
        call_id = str(item.get("call_id") or "").strip()
        output = item.get("output")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return f"Tool result {call_id}: {output}".strip()
    if item_type == "function_call":
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return f"<tool_calls><tool_call><tool_name>{name}</tool_name><parameters>{arguments}</parameters></tool_call></tool_calls>"
    content = item.get("content")
    if isinstance(content, list):
        return [dict(part) if isinstance(part, dict) else part for part in content]
    if isinstance(content, str):
        return content
    return extract_response_prompt([item]) or content or ""


def _append_response_message(messages: list[dict[str, Any]], role: object, content: object) -> None:
    if isinstance(content, str):
        if content.strip():
            messages.append({"role": str(role or "user"), "content": content.strip()})
        return
    if isinstance(content, list) and content:
        messages.append({"role": str(role or "user"), "content": content})


def messages_from_input(input_value: object, instructions: object = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_text = str(instructions or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if isinstance(input_value, str):
        if input_value.strip():
            messages.append({"role": "user", "content": input_value.strip()})
        return messages
    if isinstance(input_value, dict):
        if _is_response_content_part(input_value):
            _append_response_message(messages, "user", [dict(input_value)])
            return messages
        _append_response_message(
            messages,
            input_value.get("role") or "user",
            _message_content_from_response_item(input_value),
        )
        return messages
    if isinstance(input_value, list):
        if all(_is_response_content_part(item) for item in input_value):
            _append_response_message(messages, "user", [dict(item) for item in input_value if isinstance(item, dict)])
            return messages
        pending_parts: list[dict[str, Any]] = []
        for item in input_value:
            if _is_response_content_part(item):
                pending_parts.append(dict(item))
                continue
            if pending_parts:
                _append_response_message(messages, "user", pending_parts)
                pending_parts = []
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            _append_response_message(
                messages,
                item.get("role") or ("assistant" if item_type == "function_call" else "user"),
                _message_content_from_response_item(item),
            )
        if pending_parts:
            _append_response_message(messages, "user", pending_parts)
    return messages


def text_output_item(
    text: str,
    item_id: str | None = None,
    status: str = "completed",
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": annotations or []}],
    }


def function_call_item(name: str, arguments: dict[str, object], item_id: str | None = None, call_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id or f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": status,
        "call_id": call_id or f"call_{uuid.uuid4().hex}",
        "name": name,
        "arguments": json.dumps(arguments or {}, ensure_ascii=False),
    }


def web_search_call_item(
    query: str,
    item_id: str | None = None,
    status: str = "completed",
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "type": "search",
        "query": query,
        "queries": [query],
    }
    if sources:
        action["sources"] = [
            {"type": "url", "url": source["url"]}
            for source in sources
            if source.get("url")
        ]
    return {
        "id": item_id or f"ws_{uuid.uuid4().hex}",
        "type": "web_search_call",
        "status": status,
        "action": action,
    }


def image_output_items(prompt: str, data: list[dict[str, Any]], item_id: str | None = None) -> list[dict[str, Any]]:
    output = []
    for item in data:
        b64_json = str(item.get("b64_json") or "").strip()
        if b64_json:
            output.append({
                "id": item_id or f"ig_{len(output) + 1}",
                "type": "image_generation_call",
                "status": "completed",
                "result": b64_json,
                "revised_prompt": str(item.get("revised_prompt") or prompt).strip() or prompt,
            })
    return output


def response_created(response_id: str, model: str, created: int) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": [],
            "parallel_tool_calls": False,
        },
    }


def response_completed(
    response_id: str,
    model: str,
    created: int,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": False,
        },
    }
    if usage:
        response["response"]["usage"] = usage
    return response


def text_response_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = normalize_text_messages(normalize_messages(messages_from_input(body.get("input"), body.get("instructions"))))
    client_tools = response_client_tools(body)
    tool_prompt = build_tool_prompt(client_tools)
    if tool_prompt:
        messages.insert(0, {"role": "system", "content": tool_prompt})
    elif has_unsupported_response_tools(body):
        messages.insert(0, {"role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE})
    return model, messages


def stream_text_response(backend, body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    tools = response_client_tools(body)
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    full_text = ""
    yield response_created(response_id, model, created)
    if not tools:
        yield {"type": "response.output_item.added", "output_index": 0, "item": text_output_item("", item_id, "in_progress")}
    request = ConversationRequest(model=model, messages=messages)
    for delta in stream_text_deltas(backend, request):
        full_text += delta
        if not tools:
            yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": delta}
    output: list[dict[str, Any]] = []
    text = strip_tool_markup(full_text) if tools else full_text
    output_index = 0
    if text or not tools:
        if tools:
            yield {"type": "response.output_item.added", "output_index": output_index, "item": text_output_item("", item_id, "in_progress")}
        yield {"type": "response.output_text.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text}
        item = text_output_item(text, item_id, "completed")
        yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
        output.append(item)
        output_index += 1
    for name, arguments in parse_tool_calls(full_text):
        item = function_call_item(name, arguments, status="in_progress")
        done_item = dict(item)
        done_item["status"] = "completed"
        item["arguments"] = ""
        yield {"type": "response.output_item.added", "output_index": output_index, "item": item}
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": item["id"],
            "output_index": output_index,
            "delta": done_item["arguments"],
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": item["id"],
            "output_index": output_index,
            "arguments": done_item["arguments"],
        }
        yield {"type": "response.output_item.done", "output_index": output_index, "item": done_item}
        output.append(done_item)
        output_index += 1
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(full_text, model),
    )
    yield response_completed(response_id, model, created, output, usage)


def stream_web_search_response(body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    query = search_query_from_messages(messages) or extract_response_prompt(body.get("input"))
    if not query:
        raise HTTPException(status_code=400, detail={"error": "input text is required for web_search"})

    response_id = f"resp_{uuid.uuid4().hex}"
    search_id = f"ws_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)

    searching_item = web_search_call_item(query, search_id, "in_progress")
    yield {"type": "response.output_item.added", "output_index": 0, "item": searching_item}
    yield {"type": "response.web_search_call.in_progress", "output_index": 0, "item_id": search_id}
    yield {"type": "response.web_search_call.searching", "output_index": 0, "item_id": search_id}
    result = run_web_search(query)
    search_item = web_search_call_item(query, search_id, "completed", normalized_sources(result))
    yield {"type": "response.web_search_call.completed", "output_index": 0, "item_id": search_id}
    yield {"type": "response.output_item.done", "output_index": 0, "item": search_item}

    text, annotations = text_with_url_citations(result)
    message_item = text_output_item("", item_id, "in_progress", annotations)
    yield {"type": "response.output_item.added", "output_index": 1, "item": message_item}
    if text:
        yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 1, "content_index": 0, "delta": text}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 1, "content_index": 0, "text": text}
    message_item = text_output_item(text, item_id, "completed", annotations)
    yield {"type": "response.output_item.done", "output_index": 1, "item": message_item}
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(text, model),
    )
    yield response_completed(response_id, model, created, [search_item, message_item], usage)


def stream_image_response(
    image_outputs: Iterable[ImageOutput],
    prompt: str,
    model: str,
    input_image_tokens: int = 0,
    size: object = None,
    quality: str = "auto",
) -> Iterator[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)
    for output in image_outputs:
        if output.kind == "message":
            text = output.text
            item = text_output_item(text)
            usage = token_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_text_tokens=count_text_tokens(text, model),
            )
            yield {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text}
            yield {"type": "response.output_text.done", "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}
            yield {"type": "response.output_item.done", "output_index": 0, "item": item}
            yield response_completed(response_id, model, created, [item], usage)
            return
        if output.kind != "result":
            continue
        items = image_output_items(prompt, output.data)
        if items:
            usage = image_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_tokens=count_image_output_items_tokens(output.data, size, quality),
            )
            for output_index, item in enumerate(items):
                yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
            yield response_completed(response_id, model, created, items, usage)
            return
    raise RuntimeError("image generation failed")


def collect_response(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    completed = {}
    for event in events:
        if event.get("type") == "response.completed":
            completed = event.get("response") if isinstance(event.get("response"), dict) else {}
    if not completed:
        raise RuntimeError("response generation failed")
    return completed


def response_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if is_text_response_request(body):
        model, messages = text_response_parts(body)
        if has_web_search_tool(body) and not has_unsupported_response_tools(body):
            yield from stream_web_search_response(body, messages)
            return
        key = cache_key(body, messages, stream=bool(body.get("stream")))
        yield from chat_completion_cache.get_or_compute_stream(
            key,
            lambda: stream_text_response(text_backend(), body, messages),
        )
        return

    prompt = extract_response_prompt(body.get("input"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "input text is required"})
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    image_info = extract_response_image(body.get("input"))
    if image_info:
        image_data, mime_type = image_info
        images = encode_images([(image_data, "image.png", mime_type)])
    else:
        images = None
    input_image_tokens = count_image_content_tokens(_input_image_parts(body.get("input")), model)
    tool = response_image_tool(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        size=tool.get("size"),
        quality=str(tool.get("quality") or "auto"),
        response_format="b64_json",
        images=images,
    ))
    yield from stream_image_response(image_outputs, prompt, model, input_image_tokens, tool.get("size"), str(tool.get("quality") or "auto"))


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    events = response_events(body)
    if body.get("stream"):
        return events
    return collect_response(events)
