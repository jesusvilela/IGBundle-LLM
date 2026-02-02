
import torch
import unittest
from igbundle.nn.kan import KanLinear

class TestKanLinear(unittest.TestCase):
    def test_forward_shape(self):
        batch, in_f, out_f = 16, 8, 4
        model = KanLinear(in_f, out_f)
        x = torch.randn(batch, in_f)
        y = model(x)
        self.assertEqual(y.shape, (batch, out_f))

    def test_gradients(self):
        model = KanLinear(8, 4)
        x = torch.randn(16, 8, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check if weights received gradients
        self.assertIsNotNone(model.spline_weight.grad)
        self.assertIsNotNone(model.scale_base.grad)
        self.assertIsNotNone(model.scale_spline.grad)
        
        # Check input gradients
        self.assertIsNotNone(x.grad)

    def test_overfitting(self):
        # Can it fit a simple sin wave?
        model = KanLinear(1, 1, grid_size=10, spline_order=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x = torch.linspace(-1, 1, 100).view(-1, 1)
        y_target = torch.sin(x * 3.14)
        
        for _ in range(500):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = ((y_pred - y_target)**2).mean()
            loss.backward()
            optimizer.step()
            
        self.assertLess(loss.item(), 0.05)

if __name__ == '__main__':
    unittest.main()
