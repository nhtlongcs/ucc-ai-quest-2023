CUDA=0
FIND_LR=False

if [ ${FIND_LR} = True ]; then
    WANDB=False
else
    WANDB=True
fi
export CUDA_VISIBLE_DEVICES=${CUDA}


FOLD_INDEX=0
ucc-train -c  configs/resnet50_vigregl_deeplab.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=backbones \
              global.name=fold_${FOLD_INDEX}-resnet50_vigregl_deeplab \
              data.args.SIZE=380 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=-1 \
              trainer.lr_scheduler.name=WarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-3 \
              trainer.optimizer.args.weight_decay=5e-3

FOLD_INDEX=1
ucc-train -c  configs/resnet101_deeplab.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=backbones \
              global.name=fold_${FOLD_INDEX}-resnet101_deeplab \
              data.args.SIZE=442 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=10 \
              trainer.lr_scheduler.name=RepeatedWarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-3 \
              trainer.optimizer.args.weight_decay=1e-3

FOLD_INDEX=2
ucc-train -c  configs/resnet101_fcn.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=backbones \
              global.name=fold_${FOLD_INDEX}-resnet101_fcn \
              data.args.SIZE=540 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=10 \
              trainer.lr_scheduler.name=RepeatedWarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-3 \
              trainer.optimizer.args.weight_decay=1e-3