#!/usr/bin/env node
// Usage: node merge-gate.mjs <pr> <owner/repo>
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

const REVIEW_STEPS = ['deslop', 'no-comments', 'ponytail-review'];

/** What stops this PR from merging; empty when it may merge. */
export function problems(pr) {
  const found = [];
  if (pr.state !== 'OPEN') found.push(`the PR is ${pr.state.toLowerCase()}`);
  if (pr.isDraft) found.push('the PR is a draft');
  const review = /^## Review\s*$([\s\S]*?)(?=^## |(?![\s\S]))/m.exec(pr.body ?? '')?.[1] ?? '';
  for (const step of REVIEW_STEPS) {
    const line = new RegExp(`^- ${step}:[ \\t]*(\\S.*)$`, 'm').exec(review);
    if (!line) found.push(`"## Review" has no filled "- ${step}:" line`);
  }
  // A skipped job (one that only runs on pushes to main, say) didn't run, so it neither passes nor blocks.
  const ran = (pr.statusCheckRollup ?? []).filter((check) => check.conclusion !== 'SKIPPED');
  if (ran.length === 0) found.push('no CI check ran on the head commit');
  for (const check of ran) {
    const result = check.conclusion ?? check.state;
    if (result !== 'SUCCESS') found.push(`CI "${check.name ?? check.context}" is ${String(result).toLowerCase()}`);
  }
  return found;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [number, repo] = process.argv.slice(2);
  if (!number || !repo) {
    process.stderr.write('usage: node merge-gate.mjs <pr> <owner/repo>\n');
    process.exit(64);
  }
  const gh = (args) => execFileSync('gh', [...args, '--repo', repo], { encoding: 'utf8' });
  const pr = JSON.parse(gh(['pr', 'view', number, '--json', 'state,isDraft,body,headRefOid,statusCheckRollup']));
  const found = problems(pr);
  if (found.length > 0) {
    process.stderr.write(`Not merging #${number}:\n${found.map((p) => `- ${p}`).join('\n')}\n`);
    process.exit(1);
  }
  gh(['pr', 'merge', number, '--merge', '--match-head-commit', pr.headRefOid]);
  process.stdout.write(`Merged #${number} at ${pr.headRefOid}\n`);
}
