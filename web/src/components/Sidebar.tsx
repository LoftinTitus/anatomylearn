import { useMemo, useState } from 'react'
import type { Structure } from '../types'

interface Props {
  structures: Structure[]
  hiddenSystems: Set<string>
  selectedId: string | null
  onToggleSystem: (system: string) => void
  onSelect: (id: string) => void
}

const swatch = (s: Structure) => `rgb(${s.color.join(',')})`

export function Sidebar({ structures, hiddenSystems, selectedId, onToggleSystem, onSelect }: Props) {
  const [query, setQuery] = useState('')

  const bySystem = useMemo(() => {
    const groups = new Map<string, Structure[]>()
    for (const s of [...structures].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))) {
      groups.set(s.system, [...(groups.get(s.system) ?? []), s])
    }
    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [structures])

  const q = query.trim().toLowerCase()

  return (
    <aside className="sidebar">
      <input
        className="search"
        type="search"
        placeholder={`Search ${structures.length} structures…`}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {bySystem.map(([system, items]) => {
        const matches = q ? items.filter((s) => s.name.toLowerCase().includes(q)) : items
        if (q && matches.length === 0) return null
        const hidden = hiddenSystems.has(system)
        return (
          <details key={system} open={!!q || matches.some((s) => s.id === selectedId)}>
            <summary>
              <input
                type="checkbox"
                checked={!hidden}
                onClick={(e) => e.stopPropagation()}
                onChange={() => onToggleSystem(system)}
                aria-label={`Show ${system} system`}
              />
              <span className="system-name">{system}</span>
              <span className="count">{items.length}</span>
            </summary>
            <ul>
              {matches.map((s) => (
                <li key={s.id}>
                  <button className={s.id === selectedId ? 'selected' : ''} onClick={() => onSelect(s.id)}>
                    <span className="swatch" style={{ background: swatch(s) }} />
                    {s.name}
                  </button>
                </li>
              ))}
            </ul>
          </details>
        )
      })}
    </aside>
  )
}
