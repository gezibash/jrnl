import type { Register } from 'claude-code'

const SUBMIT_WAIT_MS = 300 // the most Enter waits; a search takes ~10 ms, a cold server ~100 ms
const RETRY_MS = 25
const MAX_QUERY = 500 // pastes can be huge; jrnl only needs the gist

// Where jrnl may be, however it was installed. `mise which` gives the real
// binary behind mise's shim. The desktop app may not have the shell's PATH,
// hence the fixed places.
const FIND = `mise=$(command -v mise || ls /opt/homebrew/bin/mise /usr/local/bin/mise "$HOME/.local/bin/mise" 2>/dev/null | head -n 1)
for c in "$("$mise" which jrnl 2>/dev/null)" "$(command -v jrnl)" \
  "$HOME/.local/bin/jrnl" /opt/homebrew/bin/jrnl /usr/local/bin/jrnl "$HOME/go/bin/jrnl" \
  "$HOME/.local/share/mise/shims/jrnl"; do
  [ -x "$c" ] && printf '%s\\t%s\\n' "$c" "$("$c" --version 2>/dev/null | head -n 1)"
done; true`

// This jrnl prints "jrnl 3.3.0" (or "jrnl dev"); 3.3.0 first serves format=context; another program named jrnl
// (homebrew-core's note taker prints "jrnl v4.6") must never be run with
// our commands: it would save them as entries.
export function servesSearch(versionLine: string): boolean {
  const m = /^jrnl (dev|(\d+)\.(\d+)\.\d+)$/.exec(versionLine.trim())
  if (!m) return false
  if (m[1] === 'dev') return true
  const [major, minor] = [Number(m[2]), Number(m[3])]
  return major > 3 || (major === 3 && minor >= 3)
}

// Prompts worth a search: a few words of the person's own, not a command.
export function searchable(text: string): string | undefined {
  const q = text.trim().replace(/\s+/g, ' ').slice(0, MAX_QUERY)
  if (q.startsWith('/') || q.startsWith('!')) return undefined
  if (q.length < 15 || q.split(' ').length < 3) return undefined
  return q
}

// The ids of the entries a <jrnl-recall> block gives.
export function entryIds(block: string): string[] {
  return [...block.matchAll(/<entry id="([0-9A-Z]{26})"/g)].map(m => m[1] as string)
}

// Silent on screen: what it did goes to the debug log (claude --debug) alone.
function debug($: { ui: { log: (text: string, options?: { to?: 'debug' }) => void } }, text: string): void {
  $.ui.log(`jrnl-recall: ${text}`, { to: 'debug' })
}

// format=context: a block written for the agent; exclude: what it already has.
function searchURL(project: string, q: string, seen: Set<string>): string {
  const exclude = seen.size ? `&exclude=${[...seen].join(',')}` : ''
  return `http://jrnl/search?project=${encodeURIComponent(project)}&limit=5&format=context${exclude}&q=${encodeURIComponent(q)}`
}

export const register: Register = on => {
  let jrnl = ''
  let project = ''
  let socket = ''
  let ready = false
  const seen = new Set<string>()

  on('session.start', async ($, e, next) => {
    try {
      const found = await $.process.run(['/bin/sh', '-c', FIND], { timeoutMs: 10_000 })
      jrnl = found.stdout.split('\n').map(l => l.split('\t')).find(([, v]) => v !== undefined && servesSearch(v))?.[0] ?? ''
      if (jrnl) {
        const where = await $.process.run([jrnl, 'where'], { timeoutMs: 5_000 })
        const dir = where.stdout.trim()
        project = dir.slice(dir.lastIndexOf('/') + 1)
        socket = `${dir.slice(0, dir.lastIndexOf('/'))}/.run/jrnl.sock`
        // `jrnl where` creates nothing; searching a missing journal would.
        ready = where.exitCode === 0 && (await $.fs.stat(`${dir}/journal.db`).then(() => true, () => false))
      }
      // Starts it detached, or replaces a server of another version; returns at once.
      if (ready) await $.process.run([jrnl, 'serve', '--background'], { timeoutMs: 10_000 })
      debug($, ready ? `using ${jrnl} for ${project}` : jrnl ? `off: no journal for ${project || e.cwd}` : 'off: no jrnl >= 3.3.0 found')
    } catch (err) {
      ready = false
      debug($, `off: ${String(err)}`)
    }
    return next(e)
  })

  on('prompt.submit', async ($, e, next) => {
    const kind = (e.origin as { kind: string } | undefined)?.kind // absent: the person's own
    const own = kind === undefined || ['composer', 'bridge', 'sdk'].includes(kind)
    const q = searchable(e.text)
    if (!ready || e.command || e.isMeta || !own || !q) return next(e)

    // The server exits after 30 idle minutes: start it again and retry
    // until the wait is spent (a cold start answers in ~100 ms).
    const t0 = await $.clock.now()
    const deadline = t0 + SUBMIT_WAIT_MS
    const search = (async () => {
      let started = false
      while ((await $.clock.now()) < deadline) {
        const r = await $.http.fetch(searchURL(project, q, seen), { socketPath: socket }).catch(() => undefined)
        if (r) return r.ok ? r.text : undefined // an error answer: retrying will not help
        if (!started) {
          started = true
          void $.process.run([jrnl, 'serve', '--background'], { timeoutMs: 10_000 }).catch(() => {})
        }
        await $.clock.sleep(RETRY_MS)
      }
      return undefined
    })()
    const text = await Promise.race([search, $.clock.sleep(SUBMIT_WAIT_MS).then(() => undefined)])

    const ids = text ? entryIds(text) : []
    debug($, text === undefined ? `no answer in ${(await $.clock.now()) - t0} ms` : `${ids.length} new in ${(await $.clock.now()) - t0} ms`)
    if (!text || ids.length === 0) return next(e)
    for (const id of ids) seen.add(id)
    return next({ ...e, context: [...(e.context ?? []), text.trim()] })
  })

  // A compacted conversation keeps only a summary: entries given before
  // may be gone from it, so they can be given again.
  on('session.compact', async ($, e, next) => {
    const done = await next(e)
    if (!e.agentId && e.trigger !== 'precompute' && 'messages' in done) {
      debug($, `compacted: ${seen.size} entries may be given again`)
      seen.clear()
    }
    return done
  })
}
