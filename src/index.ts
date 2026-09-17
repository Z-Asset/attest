/**
 * attest — 训练鉴证 skill 的 DSH 打包入口。
 *
 * 双轨单源：本包同时承载 Claude Code 插件结构（.claude-plugin/ + plugins/）
 * 与 DSH/npm 结构（package.json + src/ + cordis.patch.yml）。SKILL.md 只维护一份，
 * 住在 plugins/attest/skills/ 下，这里直接把 SKILLS_ROOT 指过去，不做任何复制。
 *
 * 与 za-analysis 的差别：attest 是 workflow skill（驱动固定脚本 pt.py，零代码生成），
 * 不派 coder/critic 子代理，所以不挂 orchestrator —— 只注册一个 skill provider。
 */
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile, readdir, stat } from 'node:fs/promises';
import type { Context } from '@deepseek-ai/cordis';
import {
  BUNDLED_SKILL_RANK,
  isSkillName,
  type SkillCandidate,
  type SkillDefinition,
  type SkillLookupOptions,
  type SkillProvider,
  type SkillProviderObservation,
} from '@deepseek-ai/dsh-skill';
import { parse as parseYaml } from 'yaml';

// ---------------------------------------------------------------------------
// Plugin identity. `name` 是 Cordis plugin id（与 cordis.patch.yml 一致）。
// ---------------------------------------------------------------------------
export const name = 'attest';
export const inject = ['skills'];

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// 双轨单源：SKILL.md 的唯一真身在 plugins/attest/skills/，Claude Code 侧也读这里。
const SKILLS_ROOT = join(PACKAGE_ROOT, 'plugins', 'attest', 'skills');

// ---------------------------------------------------------------------------
// Frontmatter 解析
// ---------------------------------------------------------------------------
interface ParsedDocument {
  data: Record<string, unknown>;
  body: string;
}

function splitFrontmatter(raw: string): ParsedDocument | undefined {
  const firstBreak = raw.indexOf('\n');
  if (firstBreak < 0) return undefined;
  if (raw.slice(0, firstBreak).replace(/\r$/, '') !== '---') return undefined;

  let lineStart = firstBreak + 1;
  let bodyStart = -1;
  while (lineStart <= raw.length) {
    const nextBreak = raw.indexOf('\n', lineStart);
    const lineEnd = nextBreak < 0 ? raw.length : nextBreak;
    if (raw.slice(lineStart, lineEnd).replace(/\r$/, '') === '---') {
      bodyStart = nextBreak < 0 ? raw.length : nextBreak + 1;
      break;
    }
    if (nextBreak < 0) return undefined;
    lineStart = nextBreak + 1;
  }
  if (bodyStart < 0) return undefined;

  const parsed = parseYaml(raw.slice(firstBreak + 1, lineStart));
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return undefined;
  return { data: parsed as Record<string, unknown>, body: raw.slice(bodyStart) };
}

const MAX_DESCRIPTION_LENGTH = 1024;

function parseSkillDocument(raw: string): { name: string; description: string; whenToUse?: string; content: string } {
  const parsed = splitFrontmatter(raw);
  if (parsed === undefined) throw new Error('missing YAML frontmatter');

  const name = typeof parsed.data.name === 'string' && parsed.data.name.length > 0 ? parsed.data.name : undefined;
  if (name === undefined) throw new Error('frontmatter requires "name"');
  if (!isSkillName(name)) throw new Error(`invalid skill name "${name}"`);

  const description = typeof parsed.data.description === 'string' && parsed.data.description.length > 0 ? parsed.data.description : undefined;
  if (description === undefined) throw new Error('frontmatter requires "description"');
  if (description.length > MAX_DESCRIPTION_LENGTH) throw new Error(`description exceeds ${MAX_DESCRIPTION_LENGTH} characters`);

  const whenToUse = typeof parsed.data.whenToUse === 'string' && parsed.data.whenToUse.length > 0 ? parsed.data.whenToUse : undefined;
  return { name, description, ...(whenToUse !== undefined ? { whenToUse } : {}), content: parsed.body.trim() };
}

// ---------------------------------------------------------------------------
// 库扫描：<root>/<name>/SKILL.md，一层深。
// ---------------------------------------------------------------------------
function isAbsentPathError(error: unknown): boolean {
  const code = typeof error === 'object' && error !== null && 'code' in error ? (error as { code?: unknown }).code : undefined;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

async function scanLibrary(root: string, signal?: AbortSignal) {
  signal?.throwIfAborted();
  let entries;
  try {
    entries = (await readdir(root, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name));
  } catch (error) {
    if (isAbsentPathError(error)) return { catalog: [], warnings: [] };
    throw error;
  }

  const catalog: { document: ReturnType<typeof parseSkillDocument>; directory: string; path: string }[] = [];
  const warnings: string[] = [];
  for (const entry of entries) {
    let kind: 'directory' | 'file' | undefined;
    if (entry.isDirectory()) kind = 'directory';
    else if (entry.isFile()) kind = 'file';
    else if (entry.isSymbolicLink()) {
      try {
        const info = await stat(join(root, entry.name));
        kind = info.isDirectory() ? 'directory' : info.isFile() ? 'file' : undefined;
      } catch {
        kind = undefined;
      }
    }

    let path: string;
    let directory: string;
    let expectedName: string;
    if (kind === 'directory') {
      path = join(root, entry.name, 'SKILL.md');
      directory = join(root, entry.name);
      expectedName = entry.name;
    } else if (kind === 'file' && entry.name.toLowerCase().endsWith('.md')) {
      path = join(root, entry.name);
      directory = root;
      expectedName = entry.name.slice(0, -'.md'.length);
    } else {
      continue;
    }

    let raw: string;
    try {
      raw = await readFile(path, { encoding: 'utf8', signal });
    } catch (error) {
      if (isAbsentPathError(error) || (typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === 'EISDIR')) continue;
      throw error;
    }

    try {
      const document = parseSkillDocument(raw);
      if (document.name !== expectedName) {
        warnings.push(`skill file ${path}: frontmatter name "${document.name}" does not match ${expectedName}; skipped`);
        continue;
      }
      catalog.push({ document, directory, path });
    } catch (error) {
      warnings.push(`skill file ${path} ignored: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { catalog, warnings };
}

// ---------------------------------------------------------------------------
// Skill provider
// ---------------------------------------------------------------------------
interface Locator {
  file: string;
  directory: string;
}

class AttestSkillProvider implements SkillProvider {
  readonly name: string;
  readonly #warn: (message: string) => void;

  constructor(providerName: string, warn: (message: string) => void) {
    this.name = providerName;
    this.#warn = warn;
  }

  async list(options: SkillLookupOptions): Promise<readonly SkillCandidate[] | SkillProviderObservation> {
    options.signal?.throwIfAborted();
    const candidates: SkillCandidate[] = [];
    const { catalog, warnings } = await scanLibrary(SKILLS_ROOT, options.signal);
    for (const w of warnings) this.#warn(w);
    for (const entry of catalog) {
      candidates.push({
        name: entry.document.name,
        description: entry.document.description,
        ...(entry.document.whenToUse !== undefined ? { whenToUse: entry.document.whenToUse } : {}),
        invocation: { modelInvocable: true, userInvocable: true },
        provider: this.name,
        source: 'bundled',
        rank: BUNDLED_SKILL_RANK,
        locator: { file: entry.path, directory: entry.directory } satisfies Locator,
        resourceBase: { kind: 'directory', path: entry.directory },
        path: entry.path,
      });
    }
    return candidates;
  }

  async get(candidate: SkillCandidate, options: SkillLookupOptions): Promise<SkillDefinition | undefined> {
    options.signal?.throwIfAborted();
    const locator = candidate.locator as Partial<Locator> | null | undefined;
    if (typeof locator?.file !== 'string' || typeof locator.directory !== 'string') return undefined;

    let raw: string;
    try {
      raw = await readFile(locator.file, { encoding: 'utf8', signal: options.signal });
    } catch (error) {
      if (isAbsentPathError(error) || (typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === 'EISDIR')) return undefined;
      throw error;
    }

    let document;
    try {
      document = parseSkillDocument(raw);
    } catch (error) {
      this.#warn(`skill file ${locator.file} ignored: ${error instanceof Error ? error.message : String(error)}`);
      return undefined;
    }
    if (document.name !== candidate.name) {
      this.#warn(`skill file ${locator.file}: frontmatter name changed to "${document.name}"; stale selection dropped`);
      return undefined;
    }

    return {
      name: document.name,
      description: document.description,
      ...(document.whenToUse !== undefined ? { whenToUse: document.whenToUse } : {}),
      invocation: { modelInvocable: true, userInvocable: true },
      provider: this.name,
      source: candidate.source,
      resourceBase: { kind: 'directory', path: locator.directory },
      path: locator.file,
      content: document.content,
    };
  }
}

export function apply(ctx: Context) {
  ctx.skills.registerProvider(
    () => new AttestSkillProvider('attest', (message) => ctx.logger?.warn(message)),
  );
}
