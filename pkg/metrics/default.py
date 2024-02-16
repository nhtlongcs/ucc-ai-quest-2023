import numpy as np
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Union, Dict
from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    default_prefix: Optional[str] = None

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
    ) -> None:
        if collect_dir is not None and collect_device != "cpu":
            raise ValueError(
                "`collec_dir` could only be configured when " "`collect_device='cpu'`"
            )

        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """

        metrics = self.compute_metrics(self.results)  # type: ignore

        # reset the results list
        self.results.clear()
        return metrics


class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.

        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        iou_metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        collect_device: str = "cpu",
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        classes: List[str] = ["background", "high_vegetation"],
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.classes = classes

    def process(self, input: Dict) -> None:
        num_classes = len(self.classes)

        pred_label = input["pred"].squeeze()

        label = input["gt"].squeeze()
        self.results.append(
            self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
        )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """

        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.classes

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        return metrics

    @staticmethod
    def intersect_and_union(
        pred_label: np.ndarray,
        label: np.ndarray,
        num_classes: int,
        ignore_index: int,
    ):
        """Calculate Intersection and Union.

        Args:
            pred_label (np.ndarray): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (np.ndarray): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            np.ndarray: The intersection of prediction and ground truth
                histogram on all classes.
            np.ndarray: The union of prediction and ground truth histogram on
                all classes.
            np.ndarray: The prediction histogram on all classes.
            np.ndarray: The ground truth histogram on all classes.
        """

        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(
            intersect.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )
        area_pred_label, _ = np.histogram(
            pred_label.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )

        area_label, _ = np.histogram(
            label.astype(float), bins=num_classes, range=(0, num_classes - 1)
        )
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | np.ndarray): The precision value.
                recall (float | np.ndarray): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [np.ndarray]: The f-score value.
            """
            score = (
                (1 + beta**2)
                * (precision * recall)
                / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f"metrics {metrics} is not supported")

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = np.array(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics


class SMAPIoUMetric(IoUMetric):
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.classes

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        for class_id, class_name in enumerate(class_names):
            for ret_metric, ret_metric_value in ret_metrics.items():
                if ret_metric == "aAcc":
                    continue
                metrics[f"{class_name}__{ret_metric}"] = np.round(ret_metric_value[class_id] * 100, 2)

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        return metrics

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        metrics: List[str] = ["mIoU"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
    ):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | np.ndarray): The precision value.
                recall (float | np.ndarray): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [np.ndarray]: The f-score value.
            """
            score = (
                (1 + beta**2)
                * (precision * recall)
                / ((beta**2 * precision) + recall)
            )
            return score

        if isinstance(metrics, str):
            metrics = [metrics]

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = np.array(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics
