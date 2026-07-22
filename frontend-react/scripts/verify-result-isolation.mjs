import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const modulePath = process.argv[2];
if (!modulePath) throw new Error('Pass the compiled resultIsolation module path.');
const { ResultRequestCoordinator } = await import(pathToFileURL(modulePath).href);

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function runOutOfOrderScenario(sequence, completionOrder) {
  const coordinator = new ResultRequestCoordinator();
  const pending = new Map();
  let visible = null;
  for (const jobId of sequence) {
    const request = coordinator.begin(jobId);
    const response = deferred();
    const key = `${jobId}:${request.generation}`;
    pending.set(key, { request, response });
    response.promise.then((result) => {
      if (coordinator.isCurrent(request) && result.jobId === request.jobId) {
        visible = structuredClone(result);
        coordinator.complete(request);
      }
    });
  }
  for (const key of completionOrder) {
    pending.get(key).response.resolve({ jobId: pending.get(key).request.jobId, frames: [key] });
    await Promise.resolve();
  }
  return visible;
}

const delayedAB = await runOutOfOrderScenario(
  ['JOB_A', 'JOB_B'],
  ['JOB_A:1', 'JOB_B:2'],
);
assert.equal(delayedAB.jobId, 'JOB_B');
assert.deepEqual(delayedAB.frames, ['JOB_B:2']);

const rapidABA = await runOutOfOrderScenario(
  ['JOB_A', 'JOB_B', 'JOB_A'],
  ['JOB_A:1', 'JOB_B:2', 'JOB_A:3'],
);
assert.equal(rapidABA.jobId, 'JOB_A');
assert.deepEqual(rapidABA.frames, ['JOB_A:3']);

const coordinator = new ResultRequestCoordinator();
const request = coordinator.begin('JOB_A');
let visible = null;
const foreign = { jobId: 'JOB_B', frames: ['foreign'] };
if (coordinator.isCurrent(request) && foreign.jobId === request.jobId) visible = foreign;
assert.equal(visible, null);

console.log('frontend result isolation scenarios passed');
