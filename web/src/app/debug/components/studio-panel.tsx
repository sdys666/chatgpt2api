"use client";

import { useMemo, useRef, useState } from "react";
import { FileText, ImageIcon, LoaderCircle, Play, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import webConfig from "@/constants/common-env";
import { getStoredAuthKey } from "@/store/auth";

type StudioMode = "md-to-text" | "md-to-image" | "md-images-to-text" | "md-images-to-image";

const modes: Array<{ value: StudioMode; title: string; desc: string; needsImage: boolean }> = [
  { value: "md-to-text", title: "MD -> 文字", desc: "消息 + MD 附件，SSE 返回文字", needsImage: false },
  { value: "md-to-image", title: "MD -> 图片", desc: "消息 + MD 附件，SSE 返回图片", needsImage: false },
  { value: "md-images-to-text", title: "MD + 图片 -> 文字", desc: "消息 + MD 附件 + 图片，SSE 返回文字", needsImage: true },
  { value: "md-images-to-image", title: "MD + 图片 -> 图片", desc: "消息 + MD 附件 + 图片，SSE 返回图片", needsImage: true },
];

function apiUrl(path: string) {
  return `${webConfig.apiUrl.replace(/\/$/, "")}${path}`;
}

function textFromChunk(value: unknown) {
  if (!value || typeof value !== "object") return "";
  const item = value as { choices?: Array<{ delta?: { content?: string }; message?: { content?: string }; text?: string }>; message?: string };
  return item.choices?.[0]?.delta?.content || item.choices?.[0]?.message?.content || item.choices?.[0]?.text || "";
}

function imagesFromChunk(value: unknown) {
  if (!value || typeof value !== "object") return [];
  const item = value as { data?: Array<{ b64_json?: string; url?: string }> };
  return (item.data || []).map((entry) => entry.b64_json ? `data:image/png;base64,${entry.b64_json}` : entry.url || "").filter(Boolean);
}

export function StudioPanel() {
  const [mode, setMode] = useState<StudioMode>("md-to-text");
  const [prompt, setPrompt] = useState("按照上传的 MD 附件，直接返回可回写的 JSON 正文。");
  const [textModel, setTextModel] = useState("gpt-5-5-thinking");
  const [imageModel, setImageModel] = useState("gpt-image-2");
  const [mdFiles, setMdFiles] = useState<File[]>([]);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [streamLog, setStreamLog] = useState("");
  const [resultText, setResultText] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const selectedMode = useMemo(() => modes.find((item) => item.value === mode) || modes[0], [mode]);

  const run = async () => {
    if (!prompt.trim()) return;
    if (selectedMode.needsImage && imageFiles.length === 0) {
      setError("这个接口需要至少上传一张图片。");
      return;
    }
    const form = new FormData();
    form.set("prompt", prompt);
    form.set("text_model", textModel);
    form.set("image_model", imageModel);
    for (const file of mdFiles) form.append("md", file, file.name);
    for (const file of imageFiles) form.append("image", file, file.name);
    setRunning(true);
    setError("");
    setStreamLog("");
    setResultText("");
    setImages([]);
    const aborter = new AbortController();
    abortRef.current = aborter;
    try {
      const token = await getStoredAuthKey();
      const response = await fetch(apiUrl(`/v1/studio/${mode}`), {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        body: form,
        signal: aborter.signal,
      });
      if (!response.ok || !response.body) {
        const message = await response.text();
        throw new Error(message || `请求失败 (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";
        for (const frame of frames) {
          if (!frame.trim()) continue;
          setStreamLog((prev) => `${prev}${frame}\n\n`);
          const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
          const raw = dataLine?.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          try {
            const parsed = JSON.parse(raw) as unknown;
            const delta = textFromChunk(parsed);
            if (delta) setResultText((prev) => prev + delta);
            const nextImages = imagesFromChunk(parsed);
            if (nextImages.length) setImages((prev) => [...prev, ...nextImages]);
          } catch {
            // Keep malformed chunks visible in the raw stream panel.
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  return (
    <div className="grid min-h-0 gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
      <section className="space-y-4 rounded-md border border-stone-200/70 bg-white p-4 dark:border-white/10 dark:bg-white/[0.03]">
        <div>
          <h2 className="text-base font-semibold text-stone-950 dark:text-white">四接口测试台</h2>
          <p className="mt-1 text-xs leading-5 text-stone-500 dark:text-stone-400">按 GPT Web 前端语义发送消息、MD 附件和图片附件，右侧实时观察 SSE。</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {modes.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => setMode(item.value)}
              className={`rounded-md border px-3 py-2 text-left text-sm transition ${mode === item.value ? "border-stone-950 bg-stone-950 text-white dark:border-white dark:bg-white dark:text-stone-950" : "border-stone-200 text-stone-600 hover:border-stone-400 dark:border-white/10 dark:text-stone-300"}`}
            >
              <div className="font-medium">{item.title}</div>
              <div className="mt-1 text-xs opacity-70">{item.desc}</div>
            </button>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>文字模型</Label>
            <Input value={textModel} onChange={(event) => setTextModel(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>图片模型</Label>
            <Input value={imageModel} onChange={(event) => setImageModel(event.target.value)} />
          </div>
        </div>
        <div className="space-y-2">
          <Label>消息</Label>
          <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} className="min-h-36" />
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex cursor-pointer flex-col gap-2 rounded-md border border-dashed border-stone-300 p-3 text-sm dark:border-white/15">
            <span className="flex items-center gap-2 font-medium"><FileText className="size-4" />MD 附件</span>
            <input type="file" accept=".md,text/markdown,text/plain" multiple className="hidden" onChange={(event) => setMdFiles(Array.from(event.target.files || []))} />
            <span className="text-xs text-stone-500">{mdFiles.length ? mdFiles.map((file) => file.name).join("，") : "点击选择 .md"}</span>
          </label>
          <label className="flex cursor-pointer flex-col gap-2 rounded-md border border-dashed border-stone-300 p-3 text-sm dark:border-white/15">
            <span className="flex items-center gap-2 font-medium"><ImageIcon className="size-4" />图片附件</span>
            <input type="file" accept="image/*" multiple className="hidden" onChange={(event) => setImageFiles(Array.from(event.target.files || []))} />
            <span className="text-xs text-stone-500">{imageFiles.length ? imageFiles.map((file) => file.name).join("，") : selectedMode.needsImage ? "此模式必选图片" : "可选"}</span>
          </label>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => void run()} disabled={running || !prompt.trim()}>
            {running ? <LoaderCircle className="animate-spin" /> : <Play />}
            开始 SSE 测试
          </Button>
          <Button variant="outline" onClick={stop} disabled={!running}>
            <Square />
            停止
          </Button>
        </div>
        {error ? <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900/60 dark:bg-rose-950/20 dark:text-rose-300">{error}</div> : null}
      </section>
      <section className="grid min-h-0 gap-4 lg:grid-cols-2">
        <div className="flex min-h-[520px] flex-col rounded-md border border-stone-200/70 bg-stone-950 text-stone-100 dark:border-white/10">
          <div className="border-b border-white/10 px-4 py-3 text-sm font-medium">SSE 原始流</div>
          <pre className="min-h-0 flex-1 overflow-auto p-4 text-xs leading-5">{streamLog || "等待 data: ...\n"}</pre>
        </div>
        <div className="flex min-h-[520px] flex-col gap-4">
          <div className="min-h-0 flex-1 rounded-md border border-stone-200/70 bg-white dark:border-white/10 dark:bg-white/[0.03]">
            <div className="border-b border-stone-200/70 px-4 py-3 text-sm font-medium dark:border-white/10">拼接文字</div>
            <pre className="max-h-[330px] overflow-auto whitespace-pre-wrap p-4 text-sm leading-6 text-stone-700 dark:text-stone-300">{resultText || "文字 delta 会在这里实时拼接。"}</pre>
          </div>
          <div className="rounded-md border border-stone-200/70 bg-white dark:border-white/10 dark:bg-white/[0.03]">
            <div className="border-b border-stone-200/70 px-4 py-3 text-sm font-medium dark:border-white/10">图片结果</div>
            <div className="grid max-h-[260px] grid-cols-2 gap-3 overflow-auto p-4">
              {images.length ? images.map((src, index) => <img key={`${src}-${index}`} src={src} alt={`result ${index + 1}`} className="w-full rounded-md border border-stone-200 object-contain dark:border-white/10" />) : <div className="col-span-2 text-sm text-stone-400">图片 chunk 会在这里显示。</div>}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
