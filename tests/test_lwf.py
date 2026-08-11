import unittest

import torch

from reid.loss.lwf import LearningWithoutForgettingLoss


class LearningWithoutForgettingLossTests(unittest.TestCase):
    def test_identical_logits_have_zero_loss(self):
        logits = torch.randn(5, 7)
        loss = LearningWithoutForgettingLoss(temperature=2.0)(
            logits,
            logits.clone(),
        )
        self.assertLess(abs(float(loss.detach())), 1e-6)

    def test_only_student_receives_gradients(self):
        student = torch.randn(4, 6, requires_grad=True)
        teacher = torch.randn(4, 6, requires_grad=True)
        loss = LearningWithoutForgettingLoss(temperature=3.0)(
            student,
            teacher,
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_shape_mismatch_is_rejected(self):
        loss_fn = LearningWithoutForgettingLoss()
        with self.assertRaises(ValueError):
            loss_fn(torch.randn(2, 3), torch.randn(2, 4))

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            LearningWithoutForgettingLoss(temperature=0.0)


if __name__ == '__main__':
    unittest.main()
