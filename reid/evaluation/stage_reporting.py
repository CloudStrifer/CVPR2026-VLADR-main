"""Readable stage results derived from evaluation history; no model or GPU use."""

import copy
import csv
import io
import json
import math
import os
import tempfile
from pathlib import Path


def stage_context(stage, total_stages):
    return dict(training_categories=sorted(v.category for v in stage.categories),
                new_categories=list(stage.new_categories), recurring_categories=list(stage.recurring_categories),
                absent_categories=list(stage.absent_categories), seen_before=list(stage.seen_before),
                total_stages=total_stages,
                training_data={v.category: dict(identities=len(v.identity_keys), images=len(v.samples),
                    source_datasets=sorted({s.source_dataset for s in v.samples})) for v in stage.categories},
                provenance='training_stream')


def _context(report, previous):
    if 'stage_context' in report:
        return copy.deepcopy(report['stage_context'])
    # Old result files contain cumulative ECPM last-update markers, so their
    # active categories can be recovered without the server's images/config.
    memory = report.get('resources', {}).get('ecpm', {})
    modes = memory.get('category_modes', {})
    if not modes:
        raise ValueError('missing stage_context and ECPM category_modes; training categories cannot be inferred')
    seen_before = set(previous[-1]['seen_categories']) if previous else set()
    active = {c for c, m in modes.items() if m['last_updated_stage'] == report['stage_id']}
    old_counts = previous[-1].get('resources', {}).get('ecpm', {}).get('categories', {}) if previous else {}
    counts = memory.get('categories', {})
    return dict(training_categories=sorted(active), new_categories=sorted(active - seen_before),
                recurring_categories=sorted(active & seen_before), absent_categories=sorted(seen_before - active),
                seen_before=sorted(seen_before), total_stages=None,
                training_data={c: dict(identities=counts[c] - old_counts.get(c, 0) if c in counts else None,
                                      images=None, source_datasets=[]) for c in active},
                provenance='inferred_from_committed_ecpm')


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_stage_results(history):
    """Keep original lifelong metrics unchanged; add an explicit old-only view.

    Forgetting is max(0, best prior score - current), in percentage points.
    New categories are null. Relative percentages have a separate name and
    are undefined for a zero historical best.
    """
    results, best = [], {}
    previous_seen = set()
    for index, report in enumerate(history):
        if report['stage_index'] != index:
            raise ValueError('stage results require a contiguous history starting at the first stage')
        if index and (any(report[k] != history[0][k] for k in
                         ('stream_fingerprint', 'split', 'routing_summary', 'beta'))
                      or set(report['retrieval']) != set(history[0]['retrieval'])):
            raise ValueError('stage results cannot mix evaluation protocols')
        context = _context(report, history[:index])
        active, seen = set(context['training_categories']), set(report['seen_categories'])
        if (not active or not active <= seen or seen != previous_seen | active
                or set(context['seen_before']) != previous_seen
                or set(context['new_categories']) != active - previous_seen
                or set(context['recurring_categories']) != active & previous_seen
                or set(context['absent_categories']) != previous_seen - active):
            raise ValueError('stage category context disagrees with evaluation history')
        result = dict(stage_id=report['stage_id'], stage_index=index, split=report['split'],
                      seen_categories=sorted(seen), stage_context=context, retrieval={},
                      routing_accuracy=report['routing']['accuracy'],
                      forgetting_unit='percentage_points', new_category_forgetting=None)
        for scope, modes in report['retrieval'].items():
            result['retrieval'][scope] = {}
            for mode in ('prototype', 'oracle'):
                raw = modes[mode]['per_category']
                if set(raw) != seen:
                    raise ValueError('evaluation must cover every seen category')
                rows = {}
                for category in sorted(seen):
                    row = dict(raw[category], status='new' if category not in previous_seen else
                               ('recurring' if category in active else 'absent'),
                               training_identities=context['training_data'].get(category, {}).get('identities', 0),
                               training_images=context['training_data'].get(category, {}).get('images', 0),
                               source_datasets=context['training_data'].get(category, {}).get('source_datasets', []),
                               previous_best={}, forgetting_pp={}, forgetting_relative_percent={})
                    for metric in ('mAP', 'Rank1'):
                        value = row[metric]
                        if not math.isfinite(value) or not 0 <= value <= 100:
                            raise ValueError('invalid percentage metric')
                        key = scope, mode, category, metric
                        prior = best.get(key)
                        drop = max(0., prior - value) if prior is not None else None
                        row['previous_best'][metric] = prior
                        row['forgetting_pp'][metric] = drop
                        row['forgetting_relative_percent'][metric] = (
                            100 * drop / prior if prior is not None and prior > 0 else None)
                        best[key] = value if prior is None else max(prior, value)
                    rows[category] = row
                result['retrieval'][scope][mode] = dict(per_category=rows,
                    macro_all_seen={m: _mean(r[m] for r in rows.values()) for m in ('mAP', 'Rank1')},
                    macro_current_training={m: _mean(rows[c][m] for c in sorted(active)) for m in ('mAP', 'Rank1')},
                    old_categories=sorted(previous_seen),
                    macro_forgetting_old_pp={m: _mean(rows[c]['forgetting_pp'][m] for c in sorted(previous_seen))
                                             for m in ('mAP', 'Rank1')})
        results.append(result)
        previous_seen = seen
    return results


def format_stage_result(result, include_oracle=False):
    def number(v):
        return '--' if v is None else '{:.2f}'.format(v)
    def names(values):
        return ', '.join(values) or '无'
    context = result['stage_context']
    total = context['total_stages']
    title = '{} 评估结果'.format(result['stage_id'])
    if total is not None:
        title += '（第 {}/{} 阶段）'.format(result['stage_index'] + 1, total)
    lines = ['## ' + title, '',
             '- 本阶段训练类别：' + names(context['training_categories']),
             '- 本阶段来源数据集：' + ('; '.join(c + ': ' + ', '.join(v['source_datasets'])
                 for c, v in sorted(context['training_data'].items()) if v.get('source_datasets')) or '旧记录未提供'),
             '- 首次出现：' + names(context['new_categories']),
             '- 再次训练：' + names(context['recurring_categories']),
             '- 已见但本阶段未训练：' + names(context['absent_categories']),
             '- 累计已见／本次评估类别：' + names(result['seen_categories']),
             '- 评估集：{}；类别路由准确率：{}%。'.format(result['split'], number(result['routing_accuracy'])), '']
    for scope, modes in result['retrieval'].items():
        for mode in (('prototype', 'oracle') if include_oracle else ('prototype',)):
            entry = modes[mode]
            lines += ['### {} / {}'.format(scope, mode), '',
                      '| 类别 | 本阶段状态 | 训练身份/图片 | mAP (%) | R1 (%) | mAP 遗忘 (pp) | R1 遗忘 (pp) |',
                      '|---|---|---:|---:|---:|---:|---:|']
            for c, r in entry['per_category'].items():
                label = dict(new='首次训练', recurring='再次训练', absent='未训练，仍评估')[r['status']]
                count = lambda v: '--' if v is None else str(v)
                lines.append('| {} | {} | {}/{} | {} | {} | {} | {} |'.format(
                    c, label, count(r['training_identities']), count(r['training_images']), number(r['mAP']),
                    number(r['Rank1']), number(r['forgetting_pp']['mAP']), number(r['forgetting_pp']['Rank1'])))
            for label, key in [('全部已见类别平均', 'macro_all_seen'), ('本阶段训练类别平均', 'macro_current_training')]:
                lines += ['', '{}：mAP={}%，R1={}%。'.format(label, number(entry[key]['mAP']), number(entry[key]['Rank1']))]
            drops = entry['macro_forgetting_old_pp']
            lines += ['', '旧类别平均遗忘：mAP={} pp，R1={} pp（仅 {} 个旧类别）。'.format(
                number(drops['mAP']), number(drops['Rank1']), len(entry['old_categories'])), '']
    lines += ['注：遗忘为历史最佳分数的非负下降量，单位是百分点；新类别及首阶段记为 --。', '']
    return '\n'.join(lines)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            temporary = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_stage_results(history, directory):
    """Regenerate derived files from committed history, never append duplicates."""
    results = build_stage_results(history)
    directory = Path(directory)
    _atomic_text(directory / 'stage_results.json', json.dumps(dict(schema_version=1, stages=results),
                 ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    _atomic_text(directory / 'stage_results.md', '# 逐阶段评估结果\n\n' + (
        '\n'.join(format_stage_result(r, include_oracle=True) for r in results) if results else '暂无已完成评估。\n'))
    output = io.StringIO(newline='')
    fields = ['stage_id', 'training_categories', 'seen_categories', 'split', 'gallery_scope', 'routing',
              'category', 'status', 'training_identities', 'training_images', 'source_datasets',
              'mAP', 'Rank1', 'mAP_forgetting_pp', 'Rank1_forgetting_pp',
              'mAP_forgetting_relative_percent', 'Rank1_forgetting_relative_percent']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        for scope, modes in result['retrieval'].items():
            for mode, entry in modes.items():
                for category, row in entry['per_category'].items():
                    writer.writerow(dict(stage_id=result['stage_id'],
                        training_categories=';'.join(result['stage_context']['training_categories']),
                        seen_categories=';'.join(result['seen_categories']), split=result['split'],
                        gallery_scope=scope, routing=mode, category=category, status=row['status'],
                        training_identities=row['training_identities'], training_images=row['training_images'],
                        source_datasets=';'.join(row['source_datasets']), mAP=row['mAP'], Rank1=row['Rank1'],
                        **{m + '_forgetting_pp': row['forgetting_pp'][m] for m in ('mAP', 'Rank1')},
                        **{m + '_forgetting_relative_percent': row['forgetting_relative_percent'][m] for m in ('mAP', 'Rank1')}))
    _atomic_text(directory / 'stage_metrics.csv', '\ufeff' + output.getvalue())
    return results
