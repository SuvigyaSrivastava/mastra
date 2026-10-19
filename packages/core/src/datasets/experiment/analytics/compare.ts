/**
 * Experiment Comparison
 *
 * Compare two experiments to detect score regressions.
 * Returns per-scorer deltas and per-item score diffs.
 */

import type { ScoreRowData } from '../../../evals/types';
import type { Mastra } from '../../../mastra';
import { computeScorerStats, isRegression } from './aggregate';
import type {
  CompareExperimentsConfig,
  ComparisonResult,
  ItemComparison,
  ScorerComparison,
  ScorerThreshold,
} from './types';

/**
 * Default threshold when not specified: no tolerance for regression.
 */
const DEFAULT_THRESHOLD: ScorerThreshold = {
  value: 0,
  direction: 'higher-is-better',
};

/**
 * Default pass threshold for computing pass rate.
 */
const DEFAULT_PASS_THRESHOLD = 0.5;

/**
 * Compare two experiments to detect score regressions.
 *
 * @param mastra - Mastra instance for storage access
 * @param config - Comparison configuration
 * @returns ComparisonResult with per-scorer and per-item comparisons
 *
 * @example
 * ```typescript
 * const result = await compareExperiments(mastra, {
 *   experimentIdA: 'baseline-experiment-id',
 *   experimentIdB: 'candidate-experiment-id',
 *   thresholds: {
 *     'accuracy': { value: 0.05, direction: 'higher-is-better' },
 *     'latency': { value: 100, direction: 'lower-is-better' },
 *   },
 * });
 *
 * if (result.hasRegression) {
 *   console.log('Quality regression detected!');
 * }
 * ```
 */
export async function compareExperiments(mastra: Mastra, config: CompareExperimentsConfig): Promise<ComparisonResult> {
  const { experimentIdA, experimentIdB, thresholds = {} } = config;
  const warnings: string[] = [];

  // 1. Get storage
  const storage = mastra.getStorage();
  if (!storage) {
    throw new Error('Storage not configured. Configure storage in Mastra instance.');
  }

  const experimentsStore = await storage.getStore('experiments');
  const scoresStore = await storage.getStore('scores');

  if (!experimentsStore) {
    throw new Error('ExperimentsStorage not configured.');
  }
  if (!scoresStore) {
    throw new Error('ScoresStorage not configured.');
  }

  // 2. Load both experiments
  const [experimentA, experimentB] = await Promise.all([
    experimentsStore.getExperimentById({ id: experimentIdA }),
    experimentsStore.getExperimentById({ id: experimentIdB }),
  ]);

  if (!experimentA) {
    throw new Error(`Experiment not found: ${experimentIdA}`);
  }
  if (!experimentB) {
    throw new Error(`Experiment not found: ${experimentIdB}`);
  }

  // 3. Check version mismatch
  const versionMismatch = experimentA.datasetVersion !== experimentB.datasetVersion;
  if (versionMismatch) {
    warnings.push(
      `Experiments have different dataset versions: ${experimentA.datasetVersion} vs ${experimentB.datasetVersion}`,
    );
  }

  // 4. Load results for both experiments
  const [resultsA, resultsB] = await Promise.all([
    experimentsStore.listExperimentResults({ experimentId: experimentIdA, pagination: { page: 0, perPage: false } }),
    experimentsStore.listExperimentResults({ experimentId: experimentIdB, pagination: { page: 0, perPage: false } }),
  ]);

  // 5. Load scores for both experiments
  const [scoresA, scoresB] = await Promise.all([
    scoresStore.listScoresByRunId({ runId: experimentIdA, pagination: { page: 0, perPage: false } }),
    scoresStore.listScoresByRunId({ runId: experimentIdB, pagination: { page: 0, perPage: false } }),
  ]);

  // 6. Handle empty experiments
  if (resultsA.results.length === 0 && resultsB.results.length === 0) {
    warnings.push('Both experiments have no results.');
    return buildEmptyResult(experimentA, experimentB, versionMismatch, warnings);
  }
  if (resultsA.results.length === 0) {
    warnings.push('Experiment A has no results.');
  }
  if (resultsB.results.length === 0) {
    warnings.push('Experiment B has no results.');
  }

  // 7. Find overlapping items
  const itemIdsA = new Set(resultsA.results.map(r => r.itemId));
  const itemIdsB = new Set(resultsB.results.map(r => r.itemId));
  const overlappingItemIds = [...itemIdsA].filter(id => itemIdsB.has(id));

  if (overlappingItemIds.length === 0) {
    warnings.push('No overlapping items between experiments.');
  }

  // 8. Group scores by scorer and item, averaging across repeated trial
  // attempts of the same item so one item contributes one effective score.
  const scoresMapA = groupScoresByScorerAndItem(scoresA.scores);
  const scoresMapB = groupScoresByScorerAndItem(scoresB.scores);

  // 9. Find all unique scorers
  const allScorerIds = new Set([...Object.keys(scoresMapA), ...Object.keys(scoresMapB)]);

  // 10. Build per-scorer comparison
  const scorers: Record<string, ScorerComparison> = {};
  let hasRegression = false;

  for (const scorerId of allScorerIds) {
    const scorerScoresA = scoresMapA[scorerId] ?? {};
    const scorerScoresB = scoresMapB[scorerId] ?? {};

    // Get scores as arrays for stats computation
    const scoresArrayA = Object.values(scorerScoresA);
    const scoresArrayB = Object.values(scorerScoresB);

    // Get threshold config for this scorer
    const thresholdConfig = thresholds[scorerId] ?? DEFAULT_THRESHOLD;
    const threshold = thresholdConfig.value;
    const direction = thresholdConfig.direction ?? 'higher-is-better';

    // Compute stats
    const statsA = computeScorerStats(scoresArrayA, DEFAULT_PASS_THRESHOLD);
    const statsB = computeScorerStats(scoresArrayB, DEFAULT_PASS_THRESHOLD);

    // Compute delta and check regression
    const delta = statsB.avgScore - statsA.avgScore;
    const regressed = isRegression(delta, threshold, direction);

    if (regressed) {
      hasRegression = true;
    }

    scorers[scorerId] = {
      statsA,
      statsB,
      delta,
      regressed,
      threshold,
    };
  }

  // 11. Build per-item comparison
  const allItemIds = new Set([...itemIdsA, ...itemIdsB]);
  const items: ItemComparison[] = [];

  for (const itemId of allItemIds) {
    const inBothExperiments = itemIdsA.has(itemId) && itemIdsB.has(itemId);

    // Build scores for this item
    const itemScoresA: Record<string, number | null> = {};
    const itemScoresB: Record<string, number | null> = {};

    for (const scorerId of allScorerIds) {
      const scoreA = scoresMapA[scorerId]?.[itemId];
      const scoreB = scoresMapB[scorerId]?.[itemId];

      itemScoresA[scorerId] = scoreA?.score ?? null;
      itemScoresB[scorerId] = scoreB?.score ?? null;
    }

    items.push({
      itemId,
      inBothExperiments,
      scoresA: itemScoresA,
      scoresB: itemScoresB,
    });
  }

  return {
    experimentA: {
      id: experimentA.id,
      datasetVersion: experimentA.datasetVersion,
    },
    experimentB: {
      id: experimentB.id,
      datasetVersion: experimentB.datasetVersion,
    },
    versionMismatch,
    hasRegression,
    scorers,
    items,
    warnings,
  };
}

/**
 * Group scores by scorer ID, then by item ID, averaging across repeated
 * trial attempts of the same item into one effective score row.
 *
 * Score rows carry no attempt number (`ScoreRowData` has no `attempt`
 * field), only `entityId` (the item ID). With repeated trials, an item can
 * have several score rows — one per attempt — that all share the same
 * `entityId`. Keying the result purely on `(scorerId, entityId)` and
 * assigning the latest row read would silently drop every attempt but one,
 * so instead every row for the same `(scorerId, entityId)` is collected and
 * averaged into a single synthetic row before being returned. Null/missing
 * scores (errored attempts) are excluded from the average, matching how
 * `computeScorerStats` already treats them as errors rather than zeros; an
 * item whose every attempt errored is reported as a single errored row.
 */
function groupScoresByScorerAndItem(scores: ScoreRowData[]): Record<string, Record<string, ScoreRowData>> {
  const grouped: Record<string, Record<string, ScoreRowData[]>> = {};

  for (const score of scores) {
    const scorerId = score.scorerId;
    const itemId = score.entityId; // entityId is the item ID for experiment scores

    grouped[scorerId] ??= {};
    grouped[scorerId]![itemId] ??= [];
    grouped[scorerId]![itemId]!.push(score);
  }

  const result: Record<string, Record<string, ScoreRowData>> = {};
  for (const [scorerId, byItem] of Object.entries(grouped)) {
    result[scorerId] = {};
    for (const [itemId, rows] of Object.entries(byItem)) {
      result[scorerId]![itemId] = averageAttempts(rows);
    }
  }

  return result;
}

/**
 * Collapse one item's attempt rows into a single row whose `score` is the
 * mean of the non-null attempt scores. The first row is used as the base
 * (for fields like `reason`/`metadata` that don't make sense to average),
 * with `score` overwritten by the computed mean. If every attempt has a
 * null/undefined score, the result keeps a null score so the row still
 * counts as an error rather than a silent zero.
 */
function averageAttempts(rows: ScoreRowData[]): ScoreRowData {
  if (rows.length === 1) {
    return rows[0]!;
  }

  const validScores = rows.map(row => row.score).filter((score): score is number => score != null);

  const averagedScore = validScores.length > 0 ? validScores.reduce((sum, s) => sum + s, 0) / validScores.length : null;

  return {
    ...rows[0]!,
    score: averagedScore as number, // ScoreRowData.score is typed non-null; see note above re: errored attempts.
  };
}

/**
 * Build an empty comparison result for edge cases.
 */
function buildEmptyResult(
  experimentA: { id: string; datasetVersion: number | null },
  experimentB: { id: string; datasetVersion: number | null },
  versionMismatch: boolean,
  warnings: string[],
): ComparisonResult {
  return {
    experimentA: {
      id: experimentA.id,
      datasetVersion: experimentA.datasetVersion,
    },
    experimentB: {
      id: experimentB.id,
      datasetVersion: experimentB.datasetVersion,
    },
    versionMismatch,
    hasRegression: false,
    scorers: {},
    items: [],
    warnings,
  };
}
