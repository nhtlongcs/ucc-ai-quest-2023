from .default import SMAPIoUMetric
from core.metrics import METRIC_REGISTRY
import torch
@METRIC_REGISTRY.register()
class SMAPIoUMetricWrapper():
    def __init__(self,label_key="labels", **kwargs):
        # super().__init__(**kwargs)
        self.label_key = label_key
        self.evaluator = SMAPIoUMetric()

    def update(self, preds, batch):
        def toNP(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy()
            return obj

        mask = toNP(batch[self.label_key])
        pred_mask = toNP(preds['msk'])
        self.evaluator.process(input={"pred": pred_mask, "gt": mask})

    def reset(self):
        pass
        # not necessary 

    def value(self):
        return self.evaluator.evaluate(0)

