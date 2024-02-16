CUDA=1
FIND_LR=False

if [ ${FIND_LR} = True ]; then
    WANDB=False
else
    WANDB=True
fi
export CUDA_VISIBLE_DEVICES=${CUDA}
PROJECT=pseudo

FOLD_INDEX=4
ucc-train -c  configs/regnety/regnety_upernet.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=${PROJECT} \
              global.name=fold_${FOLD_INDEX}-regnety_upernet \
              data.args.SIZE=380 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=-1 \
              trainer.lr_scheduler.name=WarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-4 \
              trainer.optimizer.args.weight_decay=2.5e-4 \
              trainer.num_epochs=100


FOLD_INDEX=3
ucc-train -c  configs/regnety/regnety_upernet.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=${PROJECT} \
              global.name=fold_${FOLD_INDEX}-regnety_upernet \
              data.args.SIZE=380 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=-1 \
              trainer.lr_scheduler.name=WarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-4 \
              trainer.optimizer.args.weight_decay=2.5e-4 \
              trainer.num_epochs=100 \


FOLD_INDEX=2
ucc-train -c  configs/regnety/regnety_upernet.yml \
          -o  global.find_lr=${FIND_LR} \
              global.wandb=${WANDB} \
              global.project_name=${PROJECT} \
              global.name=fold_${FOLD_INDEX}-regnety_upernet \
              data.args.SIZE=380 \
              data.args.train.fold_index=${FOLD_INDEX} \
              data.args.val.fold_index=${FOLD_INDEX} \
              model.args.freeze_epochs=-1 \
              trainer.lr_scheduler.name=WarmupPolyLR \
              trainer.optimizer.name=AdamW \
              trainer.learning_rate=1e-4 \
              trainer.optimizer.args.weight_decay=2.5e-4 \
              trainer.num_epochs=100
