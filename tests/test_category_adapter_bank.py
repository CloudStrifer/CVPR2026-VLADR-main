import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from lreid_dataset.category_stream import load_category_stream
from reid.models.category_adapter_bank import (
    CategoryAdapterBank, ReferenceConfig, build_category_model,
)
from reid.models.CLIP_ReID.model.clip.model import VisionTransformer


ROOT = Path(__file__).resolve().parents[1]


class CategoryAdapterBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.visual = VisionTransformer(2, 2, 2, 2, 8, 12, 2, 4)
        self.clean_visual = copy.deepcopy(self.visual)
        self.reference = ReferenceConfig('tiny-test-reference', input_size=(4, 4))
        self.model = CategoryAdapterBank(self.visual, self.reference, last_blocks=4, bottleneck_dim=2)
        self.images = torch.randn(3, 3, 4, 4)

    def train_person_step(self):
        self.model.add_category('person')
        self.model.add_category('vehicle')
        self.model.set_trainable_categories(['person'])
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.02, weight_decay=0.1)
        optimizer.zero_grad(set_to_none=True)
        features = self.model.encode_category(self.images, 'person')
        loss = (features - torch.randn_like(features)).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.model.adapter_parameters('person')))
        optimizer.step()
        return optimizer

    def test_initial_reference_and_zero_adapter_equal_original_projected_cls(self):
        self.model.add_category('person')
        with torch.no_grad():
            expected = self.clean_visual.eval()(self.images)[2][:, 0]
        reference = self.model.encode_reference(self.images)
        adapted = self.model.encode_category(self.images, 'person')
        self.assertEqual(tuple(reference.shape), (3, 4))
        self.assertTrue(torch.equal(expected, reference))
        self.assertTrue(torch.equal(reference, adapted))
        self.assertFalse(reference.requires_grad)

    def test_stage_stream_reuses_parameters_and_registers_only_new_categories(self):
        stream = load_category_stream(ROOT / 'config/category_progressive_example.json')
        for category in stream.stage('t1').categories:
            self.assertTrue(self.model.add_category(category.category))
        person_ids = [id(p) for p in self.model.adapter_parameters('person')]
        with torch.no_grad():
            next(self.model.adapter_parameters('person')).add_(0.5)
        snapshot = self.model.export_adapter('person')
        for category in stream.stage('t2').categories:
            created = self.model.add_category(category.category)
            self.assertEqual(created, category.category == 'panda')
        self.assertEqual(self.model.categories, ('panda', 'person', 'vehicle'))
        self.assertEqual(person_ids, [id(p) for p in self.model.adapter_parameters('person')])
        for key, value in snapshot['state'].items():
            self.assertTrue(torch.equal(value, self.model.export_adapter('person')['state'][key]))

    def test_training_changes_only_current_adapter_and_reference_stays_fixed(self):
        self.model.add_category('person')
        self.model.add_category('vehicle')
        reference_before = self.model.encode_reference(self.images)
        vehicle_before = self.model.encode_category(self.images, 'vehicle').detach()
        shared_before = {n: p.clone() for n, p in self.model.visual.state_dict().items()
                         if '.domain_adapters.' not in n}
        person_before = self.model.export_adapter('person')
        vehicle_state = self.model.export_adapter('vehicle')
        self.train_person_step()
        self.assertTrue(torch.equal(reference_before, self.model.encode_reference(self.images)))
        self.assertTrue(torch.equal(vehicle_before, self.model.encode_category(self.images, 'vehicle')))
        self.assertFalse(torch.equal(reference_before, self.model.encode_category(self.images, 'person')))
        self.model.assert_reference_unchanged()
        for key, value in shared_before.items():
            self.assertTrue(torch.equal(value, self.model.visual.state_dict()[key]))
        self.assertTrue(any(not torch.equal(v, self.model.export_adapter('person')['state'][k])
                            for k, v in person_before['state'].items()))
        for key, value in vehicle_state['state'].items():
            self.assertTrue(torch.equal(value, self.model.export_adapter('vehicle')['state'][key]))
        self.assertTrue(all(p.grad is None for n, p in self.model.visual.named_parameters()
                            if '.domain_adapters.' not in n))

    def test_two_current_categories_both_keep_gradients_across_reference_forward(self):
        self.model.add_category('person')
        self.model.add_category('vehicle')
        self.model.set_trainable_categories(['person', 'vehicle'])
        person = self.model.encode_category(self.images, 'person')
        self.model.encode_reference(self.images)
        vehicle = self.model.encode_category(self.images, 'vehicle')
        (person.square().mean() + vehicle.square().mean()).backward()
        for category in ('person', 'vehicle'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in self.model.adapter_parameters(category)))

    def test_stage_transition_clears_stale_gradients_even_with_old_optimizer(self):
        optimizer = self.train_person_step()
        person_before = self.model.export_adapter('person')
        self.model.set_trainable_categories(['vehicle'])
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in self.model.adapter_parameters('person')))
        self.model.encode_category(self.images, 'vehicle').square().mean().backward()
        optimizer.step()
        for key, value in person_before['state'].items():
            self.assertTrue(torch.equal(value, self.model.export_adapter('person')['state'][key]))

    def test_reference_is_unchanged_inside_ambient_autocast(self):
        reference = self.model.encode_reference(self.images)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            amp_reference = self.model.encode_reference(self.images)
        self.assertEqual(reference.dtype, amp_reference.dtype)
        self.assertTrue(torch.equal(reference, amp_reference))

    def test_train_eval_do_not_change_feature_definition_or_unfreeze_reference(self):
        self.train_person_step()
        self.model.train()
        train_features = self.model.encode_category(self.images, 'person').detach()
        train_reference = self.model.encode_reference(self.images)
        self.model.eval()
        self.assertTrue(torch.equal(train_features, self.model.encode_category(self.images, 'person')))
        self.assertTrue(torch.equal(train_reference, self.model.encode_reference(self.images)))
        self.assertFalse(self.model.visual.training)
        self.assertTrue(all(not p.requires_grad for n, p in self.model.visual.named_parameters()
                            if '.domain_adapters.' not in n))

    def test_shared_bn_running_state_remains_frozen_under_parent_train(self):
        visual = copy.deepcopy(self.clean_visual)
        visual.shared_probe = nn.BatchNorm1d(4)
        model = CategoryAdapterBank(visual, self.reference, bottleneck_dim=2)
        model.add_category('person')
        model.set_trainable_categories(['person'])
        parent = nn.ModuleList([model])
        parent.train()
        before = visual.shared_probe.running_mean.clone()
        visual.shared_probe(torch.randn(3, 4))
        self.assertFalse(visual.shared_probe.training)
        self.assertTrue(torch.equal(before, visual.shared_probe.running_mean))
        model.assert_reference_unchanged()

    def test_input_gradient_reference_disabled_student_preserved(self):
        self.model.add_category('person')
        self.model.set_trainable_categories(['person'])
        images = self.images.clone().requires_grad_()
        self.assertFalse(self.model.encode_reference(images).requires_grad)
        self.model.encode_category(images, 'person').square().mean().backward()
        self.assertIsNotNone(images.grad)

    def test_active_adapter_restored_after_reference_and_forward_error(self):
        self.model.add_category('person')
        self.model.add_category('vehicle')
        key = self.model.category_key('vehicle')
        self.model.visual.set_active_domain_adapter(key)
        self.model.encode_reference(self.images)
        self.assertEqual(self.model.visual.get_active_domain_adapter(), key)
        with self.assertRaisesRegex(ValueError, 'image size'):
            self.model.encode_category(torch.randn(1, 3, 6, 6), 'person')
        self.assertEqual(self.model.visual.get_active_domain_adapter(), key)

    def test_unicode_and_punctuation_category_keys_are_stable_and_safe(self):
        another = CategoryAdapterBank(copy.deepcopy(self.clean_visual), self.reference, bottleneck_dim=2)
        categories = ['giant.panda/亚洲', 'person', 'Person']
        for category in categories:
            self.model.add_category(category)
        for category in reversed(categories):
            another.add_category(category)
            self.assertEqual(self.model.category_key(category), another.category_key(category))
            self.assertNotIn('.', self.model.category_key(category))
            self.assertNotIn('/', self.model.category_key(category))
        self.assertEqual(len(set(self.model._category_keys.values())), 3)

    def test_copy_independent_parameters_and_snapshot_not_live(self):
        self.train_person_step()
        snapshot = self.model.export_adapter('person')
        self.model.copy_category('person', 'panda')
        source = list(self.model.adapter_parameters('person'))
        target = list(self.model.adapter_parameters('panda'))
        self.assertTrue(all(torch.equal(p, q) and p.data_ptr() != q.data_ptr() for p, q in zip(source, target)))
        self.model.set_trainable_categories(['panda'])
        optimizer = torch.optim.SGD(self.model.trainable_parameters(), lr=0.2)
        self.model.encode_category(self.images, 'panda').square().mean().backward()
        optimizer.step()
        for key, value in snapshot['state'].items():
            self.assertTrue(torch.equal(value, self.model.export_adapter('person')['state'][key]))
        with self.assertRaises(ValueError):
            self.model.copy_category('person', 'panda')
        with torch.no_grad():
            source[0].add_(1)
        self.assertFalse(torch.equal(source[0], snapshot['state'][next(iter(snapshot['state']))]))

    def test_malformed_snapshot_rejected_before_category_creation(self):
        self.model.add_category('person')
        for failure in ('shape', 'nan', 'missing', 'reference', 'scale'):
            snapshot = self.model.export_adapter('person')
            first = next(iter(snapshot['state']))
            if failure == 'shape':
                snapshot['state'][first] = torch.zeros(1)
            elif failure == 'nan':
                snapshot['state'][first].fill_(float('nan'))
            elif failure == 'missing':
                del snapshot['state'][first]
            elif failure == 'reference':
                snapshot['reference_signature'] = 'wrong'
            else:
                snapshot['adapter_config']['scale'] = 2.0
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.model.load_adapter('panda', snapshot)
            self.assertNotIn('panda', self.model.categories)

    def test_bank_roundtrip_reconstructs_names_and_features(self):
        self.train_person_step()
        snapshot = self.model.export_bank()
        restored = CategoryAdapterBank(copy.deepcopy(self.clean_visual), self.reference, bottleneck_dim=2)
        restored.load_bank(snapshot)
        self.assertEqual(restored.categories, self.model.categories)
        self.assertEqual(restored.trainable_categories, ())
        for category in restored.categories:
            self.assertTrue(torch.equal(restored.encode_category(self.images, category),
                                        self.model.encode_category(self.images, category)))
        with self.assertRaises(ValueError):
            restored.load_bank(snapshot)

    def test_bank_rejects_same_architecture_different_weights_or_preprocessing(self):
        self.model.add_category('person')
        snapshot = self.model.export_bank()
        for change in ('weights', 'preprocessing'):
            visual = copy.deepcopy(self.clean_visual)
            reference = self.reference
            if change == 'weights':
                with torch.no_grad():
                    visual.proj.add_(0.1)
            else:
                reference = ReferenceConfig('tiny-test-reference', (4, 4), resize_mode='stretch')
            restored = CategoryAdapterBank(visual, reference, bottleneck_dim=2)
            with self.subTest(change=change), self.assertRaises(ValueError):
                restored.load_bank(snapshot)

    def test_bank_validates_all_categories_before_mutating_registry(self):
        self.model.add_category('person')
        self.model.add_category('vehicle')
        for change in ('source', 'keys', 'tensor', 'metadata'):
            snapshot = self.model.export_bank()
            if change == 'source':
                snapshot['adapters']['vehicle']['source_category'] = 'person'
            elif change == 'keys':
                snapshot['category_keys']['vehicle'] = snapshot['category_keys']['person']
            elif change == 'metadata':
                snapshot['reference']['weight_id'] = 'another-weight'
            else:
                first = next(iter(snapshot['adapters']['vehicle']['state']))
                snapshot['adapters']['vehicle']['state'][first].fill_(float('nan'))
            restored = CategoryAdapterBank(copy.deepcopy(self.clean_visual), self.reference, bottleneck_dim=2)
            with self.subTest(change=change), self.assertRaises(ValueError):
                restored.load_bank(snapshot)
            self.assertEqual(restored.categories, ())

    def test_full_checkpoint_tensor_serialization_and_raw_model_restore(self):
        self.train_person_step()
        self.model.create_temporary_head('person', 2)
        checkpoint = self.model.export_checkpoint()
        self.assertFalse(any('temporary' in key or 'domain_adapters' in key
                             for key in checkpoint['reference_state']))
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        restored = CategoryAdapterBank.from_checkpoint(torch.load(buffer, weights_only=True))
        self.assertEqual(len(restored.temporary_heads), 0)
        self.assertFalse(restored.training)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        self.assertTrue(torch.equal(self.model.encode_reference(self.images), restored.encode_reference(self.images)))
        self.assertTrue(torch.equal(self.model.encode_category(self.images, 'person'),
                                    restored.encode_category(self.images, 'person')))

    def test_temporary_classifier_is_current_only_and_discardable(self):
        self.model.add_category('person')
        self.model.set_trainable_categories(['person'])
        head = self.model.create_temporary_head('person', 2)
        self.assertEqual((head.in_features, head.out_features), (4, 2))
        self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in self.model.modules()))
        logits = self.model.classify(self.model.encode_category(self.images, 'person'), 'person')
        nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertEqual(tuple(logits.shape), (3, 2))
        with self.assertRaises(ValueError):
            self.model.create_temporary_head('person', 3)
        self.model.discard_temporary_heads()
        self.assertEqual(self.model.create_temporary_head('person', 3).out_features, 3)

    def test_reference_mutation_is_detected_before_export(self):
        with torch.no_grad():
            self.model.visual.proj.add_(0.1)
        with self.assertRaisesRegex(RuntimeError, 'reference weights/buffers changed'):
            self.model.export_bank()

    def test_frozen_preprocessing_factory_matches_config(self):
        from PIL import Image

        _, fixed = self.model.make_transforms(crop_padding=0)
        image = Image.new('RGB', (6, 4), (20, 50, 100))
        self.assertEqual(tuple(fixed(image).shape), (3, 4, 4))
        self.assertTrue(torch.equal(fixed(image), fixed(image)))
        with self.assertRaises(ValueError):
            self.model.make_transforms(height=8)

    def test_unknown_category_does_not_auto_create_or_change_training_policy(self):
        with self.assertRaises(KeyError):
            self.model.encode_category(self.images, 'panda')
        with self.assertRaises(KeyError):
            self.model.set_trainable_categories(['panda'])
        self.assertEqual(self.model.categories, ())
        with self.assertRaises(TypeError):
            self.model.set_trainable_categories('person')

    def test_invalid_architecture_and_input_contract(self):
        with self.assertRaises(ValueError):
            CategoryAdapterBank(VisionTransformer(2, 2, 2, 2, 8, 4, 2, 4), self.reference)
        for options in ({'last_blocks': 0}, {'last_blocks': 1.5}, {'bottleneck_dim': 0}, {'scale': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CategoryAdapterBank(copy.deepcopy(self.clean_visual), self.reference, **options)
        with self.assertRaises(TypeError):
            self.model.encode_reference(torch.zeros(1, 3, 4, 4, dtype=torch.uint8))

    def test_local_factory_missing_weight_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                build_category_model(Path(directory) / 'not_here.pt')

    def test_wrapper_new_entry_does_not_use_legacy_model_constructor(self):
        from reid.models.wrapper import make_category_model

        with patch('reid.models.category_adapter_bank.build_category_model', return_value=self.model) as factory:
            self.assertIs(make_category_model(device='cpu'), self.model)
            factory.assert_called_once_with(device='cpu')


if __name__ == '__main__':
    unittest.main()
