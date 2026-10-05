import { marked } from 'marked'

/**
 * Learning content, one Markdown file per topic in /content/structures.
 * Frontmatter `ids` lists the structure ids (from structures.json) the file covers,
 * so paired structures (left/right kidney) can share one page.
 */
export interface ContentPage {
  title: string
  ids: string[]
  level: string
  reviewed: boolean
  html: string
}

const files = import.meta.glob('../../content/structures/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

function parse(raw: string): ContentPage {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  const meta: Record<string, string> = {}
  const body = match ? match[2] : raw
  for (const line of (match?.[1] ?? '').split('\n')) {
    const i = line.indexOf(':')
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim()
  }
  return {
    title: meta.title ?? '',
    ids: (meta.ids ?? '').split(',').map((s) => s.trim()).filter(Boolean),
    level: meta.level ?? 'intro',
    reviewed: meta.reviewed === 'true',
    html: marked.parse(body, { async: false }),
  }
}

const byId = new Map<string, ContentPage>()
for (const raw of Object.values(files)) {
  const page = parse(raw)
  for (const id of page.ids) byId.set(id, page)
}

export const contentFor = (id: string) => byId.get(id)
