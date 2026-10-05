import { contentFor } from '../content'
import type { Manifest, Structure } from '../types'

export function InfoPanel({ structure, manifest }: { structure: Structure | null; manifest: Manifest }) {
  if (!structure) {
    return (
      <section className="info empty">
        <div>
          <p>Click a structure in the 3D model, the CT slices, or the list to learn about it.</p>
          <p className="note">
            Scan:{' '}
            {manifest.sourceUrl ? (
              <a href={manifest.sourceUrl} target="_blank" rel="noreferrer">
                {manifest.source}
              </a>
            ) : (
              manifest.source
            )}
          </p>
          {manifest.noData.map((gap) => (
            <p key={gap.zRange.join()} className="note">
              {gap.note}.
            </p>
          ))}
        </div>
      </section>
    )
  }
  const page = contentFor(structure.id)
  return (
    <section className="info">
      <header>
        <h2>{structure.name}</h2>
        <div className="chips">
          <span className="chip">{structure.system}</span>
          {structure.pathology && <span className="chip warn">pathology</span>}
          {structure.snomed && <span className="chip muted" title={structure.snomed.meaning}>SNOMED {structure.snomed.code}</span>}
          <span className="chip muted">{structure.volumeMl} mL in this scan</span>
        </div>
        {structure.truncated && <p className="note">Partly outside the scanned region, so the model shows only part of it.</p>}
      </header>
      {page ? (
        <article>
          {!page.reviewed && <p className="note draft">Draft content. Not yet reviewed by a clinician.</p>}
          <div dangerouslySetInnerHTML={{ __html: page.html }} />
        </article>
      ) : (
        <p className="muted">
          No learning content yet. Add <code>content/structures/{structure.id}.md</code> with <code>ids: {structure.id}</code>.
        </p>
      )}
    </section>
  )
}
