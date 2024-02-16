CUDA=0
FIND_LR=False

if [ ${FIND_LR} = True ]; then
    WANDB=False
else
    WANDB=True
fi
export CUDA_VISIBLE_DEVICES=${CUDA}
PROJECT=pseudo

FOLD_INDEX=0
ucc-train   -c  configs/segformer/segformerb3-ade.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-segformer-b3-ade \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=1e-4 \
                trainer.optimizer.args.weight_decay=1e-2


FOLD_INDEX=1
ucc-train   -c  configs/segformer/segformerb4-ade.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-segformer-b4-ade \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=1e-4 \
                trainer.optimizer.args.weight_decay=1e-2

FOLD_INDEX=2
ucc-train   -c  configs/segformer/segformerb5-ade.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-segformer-b5-ade \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=1e-4 \
                trainer.optimizer.args.weight_decay=1e-2


