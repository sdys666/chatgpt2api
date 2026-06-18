from __future__ import annotations

import base64
import mimetypes

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.editable_file_task_service import editable_file_task_service
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    openai_search,
)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class EditableFileTaskRequest(BaseModel):
    prompt: str = ""
    base64_images: list[str] = Field(default_factory=list)
    client_task_id: str | None = None


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def _clean_form_text(value: object, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text or default


async def _studio_form_payload(request: Request) -> tuple[dict[str, object], list[dict[str, object]], list[tuple[bytes, str, str]]]:
    form = await request.form()
    prompt = _clean_form_text(form.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    payload: dict[str, object] = {
        "prompt": prompt,
        "text_model": _clean_form_text(form.get("text_model") or form.get("model"), "auto"),
        "image_model": _clean_form_text(form.get("image_model") or form.get("model"), "gpt-image-2"),
        "n": int(_clean_form_text(form.get("n"), "1")),
        "size": _clean_form_text(form.get("size")),
        "quality": _clean_form_text(form.get("quality"), "auto"),
    }
    attachments: list[dict[str, object]] = []
    images: list[tuple[bytes, str, str]] = []
    for key, value in form.multi_items():
        if not isinstance(value, UploadFile):
            continue
        data = await value.read()
        if not data:
            continue
        filename = value.filename or ("image.png" if str(value.content_type or "").startswith("image/") else "attachment.md")
        mime_type = value.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        item = {
            "filename": filename,
            "mime_type": mime_type,
            "base64": base64.b64encode(data).decode("ascii"),
        }
        if key in {"md", "md[]", "attachment", "attachments"} or filename.lower().endswith(".md") or mime_type in {"text/markdown", "text/plain"}:
            attachments.append(item)
            continue
        if key in {"image", "image[]", "images", "images[]"} or mime_type.startswith("image/"):
            images.append((data, filename, mime_type))
    return payload, attachments, images


def _studio_text_body(payload: dict[str, object], attachments: list[dict[str, object]], images: list[tuple[bytes, str, str]]) -> dict[str, object]:
    merged_attachments = list(attachments)
    for data, filename, mime_type in images:
        merged_attachments.append({
            "filename": filename,
            "mime_type": mime_type,
            "base64": base64.b64encode(data).decode("ascii"),
        })
    return {
        "model": payload["text_model"],
        "messages": [{"role": "user", "content": payload["prompt"]}],
        "attachments": merged_attachments,
        "stream": True,
    }


def _studio_image_body(payload: dict[str, object], attachments: list[dict[str, object]], request: Request, images: list[tuple[bytes, str, str]] | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "model": payload["image_model"],
        "prompt": payload["prompt"],
        "attachments": attachments,
        "n": payload["n"],
        "size": payload["size"] or None,
        "quality": payload["quality"],
        "response_format": "b64_json",
        "stream": True,
        "base_url": resolve_image_base_url(request),
    }
    if images is not None:
        body["images"] = images
    return body


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_v1_image_generations.handle, payload)

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        payload["images"] = await read_image_sources(image_sources)
        if mask_sources:
            payload["mask"] = await read_image_sources(mask_sources)
        payload["base_url"] = resolve_image_base_url(request)
        return await call.run(openai_v1_image_edit.handle, payload)

    @router.post("/v1/chat/completions")
    async def create_chat_completion(body: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(
            identity,
            "/v1/chat/completions",
            model,
            "文本生成",
            request_text=request_preview,
            request_shape=request_shape(payload.get("messages")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/studio/md-to-text")
    async def studio_md_to_text(request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload, attachments, _images = await _studio_form_payload(request)
        body = _studio_text_body(payload, attachments, [])
        call = LoggedCall(identity, "/v1/chat/studio-md-to-text", str(body["model"]), "MD返回文字", request_text=str(payload["prompt"]))
        await filter_or_log(call, str(payload["prompt"]))
        return await call.run(openai_v1_chat_complete.handle, body)

    @router.post("/v1/studio/md-to-image")
    async def studio_md_to_image(request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload, attachments, _images = await _studio_form_payload(request)
        body = _studio_image_body(payload, attachments, request)
        call = LoggedCall(identity, "/v1/images/studio-md-to-image", str(body["model"]), "MD返回图片", request_text=str(payload["prompt"]))
        await filter_or_log(call, str(payload["prompt"]))
        return await call.run(openai_v1_image_generations.handle, body)

    @router.post("/v1/studio/md-images-to-text")
    async def studio_md_images_to_text(request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload, attachments, images = await _studio_form_payload(request)
        if not images:
            raise HTTPException(status_code=400, detail={"error": "image is required"})
        body = _studio_text_body(payload, attachments, images)
        call = LoggedCall(identity, "/v1/chat/studio-md-images-to-text", str(body["model"]), "MD+图片返回文字", request_text=str(payload["prompt"]))
        await filter_or_log(call, str(payload["prompt"]))
        return await call.run(openai_v1_chat_complete.handle, body)

    @router.post("/v1/studio/md-images-to-image")
    async def studio_md_images_to_image(request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload, attachments, images = await _studio_form_payload(request)
        if not images:
            raise HTTPException(status_code=400, detail={"error": "image is required"})
        body = _studio_image_body(payload, attachments, request, images)
        call = LoggedCall(identity, "/v1/images/studio-md-images-to-image", str(body["model"]), "MD+图片返回图片", request_text=str(payload["prompt"]))
        await filter_or_log(call, str(payload["prompt"]))
        return await call.run(openai_v1_image_edit.handle, body)

    @router.post("/v1/responses")
    async def create_response(body: ResponseCreateRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses",
            model,
            "Responses",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        return await run_in_threadpool(editable_file_task_service.list_tasks, identity, task_ids)

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await run_in_threadpool(editable_file_task_service.public_file_path, file_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_ppt,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_psd,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    return router
