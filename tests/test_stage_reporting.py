import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from reid.evaluation.stage_reporting import build_stage_results, format_stage_result, write_stage_results


def history_fixture():
    stages = [('person',), ('person', 'panda'), ('vehicle', 'panda'), ('tiger', 'vehicle'), ('boat', 'person')]
    scores = [dict(person=80), dict(person=70, panda=50), dict(person=60, panda=55, vehicle=40),
              dict(person=65, panda=54, vehicle=42, tiger=75),
              dict(person=75, panda=54, vehicle=41, tiger=74, boat=90)]
    reports, seen = [], set()
    for i, (active, values) in enumerate(zip(stages, scores)):
        active = set(active)
        context = dict(training_categories=sorted(active), seen_before=sorted(seen),
                       new_categories=sorted(active - seen), recurring_categories=sorted(active & seen),
                       absent_categories=sorted(seen - active), total_stages=5,
                       training_data={c: dict(identities=8, images=32, source_datasets=[c + '_dataset']) for c in active},
                       provenance='fixture')
        seen |= active
        metric = {c: dict(mAP=v, Rank1=v + 5) for c, v in values.items()}
        reports.append(dict(stage_index=i, stage_id='t' + str(i + 1), seen_categories=sorted(seen),
            stage_context=context, split='test', stream_fingerprint='fixture', routing_summary='ecpm', beta=.5,
            routing=dict(accuracy=99.), retrieval={scope: {mode: dict(per_category=copy.deepcopy(metric))
                for mode in ('prototype', 'oracle')} for scope in ('per_dataset', 'mixed')}))
    return reports


class StageReportingTests(unittest.TestCase):
    def test_five_stages_show_active_and_absent_categories_without_future_leak(self):
        results = build_stage_results(history_fixture())
        self.assertEqual(len(results), 5)
        third = results[2]
        self.assertEqual(third['stage_context']['training_categories'], ['panda', 'vehicle'])
        self.assertEqual(third['seen_categories'], ['panda', 'person', 'vehicle'])
        rows = third['retrieval']['per_dataset']['prototype']['per_category']
        self.assertEqual(rows['person']['status'], 'absent')
        self.assertEqual(rows['person']['training_images'], 0)
        self.assertEqual(rows['vehicle']['status'], 'new')
        self.assertNotIn('boat', rows)
        text = format_stage_result(third)
        self.assertIn('第 3/5 阶段', text)
        self.assertIn('本阶段训练类别：panda, vehicle', text)
        self.assertIn('未训练，仍评估', text)

    def test_forgetting_uses_best_prior_excludes_new_and_separates_relative_rate(self):
        results = build_stage_results(history_fixture())
        first = results[0]['retrieval']['per_dataset']['prototype']
        self.assertIsNone(first['macro_forgetting_old_pp']['mAP'])
        second = results[1]['retrieval']['per_dataset']['prototype']
        self.assertEqual(second['macro_forgetting_old_pp']['mAP'], 10.)
        self.assertEqual(second['per_category']['person']['forgetting_relative_percent']['mAP'], 12.5)
        self.assertIsNone(second['per_category']['panda']['forgetting_pp']['mAP'])
        third = results[2]['retrieval']['per_dataset']['prototype']
        self.assertEqual(third['per_category']['person']['forgetting_pp']['mAP'], 20.)
        self.assertEqual(third['per_category']['panda']['forgetting_pp']['mAP'], 0.)
        self.assertEqual(third['macro_forgetting_old_pp']['mAP'], 10.)
        self.assertEqual(third['macro_current_training']['mAP'], 47.5)
        self.assertAlmostEqual(third['macro_all_seen']['mAP'], 155/3)
        last = results[-1]['retrieval']['per_dataset']['prototype']
        self.assertEqual(last['per_category']['person']['forgetting_pp']['mAP'], 5.)

    def test_zero_best_relative_forgetting_is_undefined(self):
        history = history_fixture()
        history[0]['retrieval']['per_dataset']['prototype']['per_category']['person']['mAP'] = 0
        result = build_stage_results(history)[1]['retrieval']['per_dataset']['prototype']['per_category']['person']
        self.assertEqual(result['forgetting_pp']['mAP'], 0.)
        self.assertIsNone(result['forgetting_relative_percent']['mAP'])

    def test_exports_are_deterministic_and_rewritten_from_authoritative_prefix(self):
        history = history_fixture()
        original = copy.deepcopy(history)
        with tempfile.TemporaryDirectory() as folder:
            write_stage_results(history, folder)
            paths = list(Path(folder).iterdir())
            before = {p.name: p.read_bytes() for p in paths}
            write_stage_results(history, folder)
            self.assertEqual(before, {p.name: p.read_bytes() for p in paths})
            write_stage_results(history[:2], folder)
            data = json.loads((Path(folder) / 'stage_results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['stages']), 2)
            with (Path(folder) / 'stage_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 12)  # (one + two categories) * two scopes * two routes
            self.assertEqual({r['stage_id'] for r in rows}, {'t1', 't2'})
        self.assertEqual(history, original)

    def test_legacy_context_recovery_and_unknown_images_are_explicit(self):
        history = history_fixture()[:2]
        history[0].pop('stage_context')
        history[0]['resources'] = dict(ecpm=dict(categories=dict(person=8),
            category_modes=dict(person=dict(last_updated_stage='t1'))))
        history[1].pop('stage_context')
        history[1]['resources'] = dict(ecpm=dict(categories=dict(person=16, panda=8),
            category_modes=dict(person=dict(last_updated_stage='t2'), panda=dict(last_updated_stage='t2'))))
        result = build_stage_results(history)[1]
        self.assertIsNone(result['stage_context']['total_stages'])
        person = result['stage_context']['training_data']['person']
        self.assertEqual(person['identities'], 8)
        self.assertIsNone(person['images'])

    def test_missing_categories_context_or_changed_protocol_rejected(self):
        for kind in ('context', 'protocol', 'coverage', 'order'):
            history = history_fixture()
            if kind == 'context':
                history[1]['stage_context']['training_categories'] = ['panda']
            elif kind == 'protocol':
                history[2]['split'] = 'validation'
            elif kind == 'coverage':
                del history[2]['retrieval']['mixed']['oracle']['per_category']['person']
            else:
                history[2]['stage_index'] = 8
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build_stage_results(history)


if __name__ == '__main__':
    unittest.main()
