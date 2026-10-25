import unittest

import numpy as np
import pandas as pd

from tools.recompute_frozen_results import recompute


def exported():
    rows = []
    for seed in [42, 43]:
        for rid, target, experts in [('a', 100., [90., 110., 200.]), ('b', 200., [150., 210., 300.])]:
            for gate, weights in [('soft', [.5, .5, 0.]), ('hard', [0., 1., 0.])]:
                row = dict(model='BiLSTM', gate=gate, seed=seed, record_id=rid,
                           observed_cycle=1000., target=target)
                row.update({f'expert{i}': e for i, e in enumerate(experts)})
                row.update({f'weight{i}': w for i, w in enumerate(weights)})
                rows.append(row)
    return pd.DataFrame(rows)


class FrozenResultsContracts(unittest.TestCase):
    def test_per_seed_metrics_and_shuffled_identity(self):
        rows, metrics, summary = recompute(exported().sample(frac=1, random_state=3))
        self.assertEqual(len(rows), 8)
        soft = metrics[metrics.gate == 'soft']
        np.testing.assert_allclose(soft.mae, 10.)
        np.testing.assert_allclose(soft.rmse, np.sqrt(200.))
        self.assertEqual(summary[summary.gate == 'soft'].iloc[0].seeds, 2)

    def test_rejects_weight_errors_and_duplicate_rows(self):
        frame = exported()
        frame.loc[0, 'weight0'] = 1.
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            recompute(frame)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            recompute(pd.concat([exported(), exported().iloc[:1]]))

    def test_rejects_changed_targets_or_expert_outputs(self):
        for column in ['target', 'expert0']:
            frame = exported()
            frame.loc[0, column] += 1
            with self.assertRaisesRegex(ValueError, 'preserve targets'):
                recompute(frame)

    def test_rejects_missing_observation_and_mc_means(self):
        with self.assertRaisesRegex(ValueError, 'observation identities'):
            recompute(exported().iloc[1:])
        frame = exported()
        frame['prediction_type'] = 'MC'
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            recompute(frame)

    def test_rejects_stored_prediction_mismatch(self):
        frame = exported()
        frame['prediction'] = 0.
        with self.assertRaisesRegex(ValueError, 'Stored predictions'):
            recompute(frame)

    def test_rejects_entire_missing_seed_for_one_gate(self):
        frame = exported()
        frame = frame[~((frame.seed == 43) & (frame.gate == 'soft'))]
        with self.assertRaisesRegex(ValueError, 'matching seed sets'):
            recompute(frame)


if __name__ == '__main__':
    unittest.main()
