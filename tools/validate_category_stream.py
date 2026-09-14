"""Audit a category-interleaved stream without importing torch or opening images."""

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lreid_dataset.category_stream import StreamProtocolError, load_category_stream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    parser.add_argument('--check-images', action='store_true',
                        help='also check image file existence; never decode images')
    parser.add_argument('--output', help='optional JSON audit report')
    args = parser.parse_args(argv)
    try:
        stream = load_category_stream(args.stream_config, check_images=args.check_images)
        report = stream.audit_report()
        report['image_existence_checked'] = args.check_images
        if args.output:
            output = Path(args.output).expanduser().resolve()
            if output == Path(args.stream_config).expanduser().resolve():
                raise StreamProtocolError('--output cannot overwrite the stream config')
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + '.tmp')
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            os.replace(str(temporary), str(output))
    except (StreamProtocolError, OSError) as exc:
        print('INVALID: {}'.format(exc), file=sys.stderr)
        return 2

    for stage in report['stages']:
        print('{} | new={} recurring={} absent={}'.format(
            stage['stage_id'], stage['new_categories'], stage['recurring_categories'],
            stage['absent_categories']))
        for category in stage['categories']:
            print('  {}: {} identities, {} images'.format(
                category['category'], category['identities'], category['images']))
    print('VALID: {} stages, {} training identities, {} training images, {} evaluation sets'.format(
        len(report['stages']), report['total_train_identities'], report['total_train_images'],
        len(report['evaluation'])))
    print('Image existence checked: {}; image contents read: False'.format(args.check_images))
    print('Protocol fingerprint: {}'.format(stream.fingerprint))
    for warning in report['warnings']:
        print('NOTE: {}'.format(warning))
    if args.output:
        print('Report: {}'.format(Path(args.output).resolve()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
