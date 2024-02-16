CUDA=0
FIND_LR=True

if [ ${FIND_LR} = True ]; then
    WANDB=False
else
    WANDB=True
fi
export CUDA_VISIBLE_DEVICES=${CUDA}
PROJECT=pseudo

FOLD_INDEX=1
ucc-train   -c  configs/mask2former/mask2former-ade-small.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-mask2former-ade-small \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=5e-5 \
                trainer.optimizer.args.weight_decay=1e-3 \
                loss.smooth_factor=0.3 \
                global.SEED=3407

FOLD_INDEX=2
ucc-train   -c  configs/mask2former/mask2former-coco-21k.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-mask2former-coco-21k \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=1e-4 \
                trainer.optimizer.args.weight_decay=1e-2 \
                loss.smooth_factor=0.3 \
                global.SEED=10

FOLD_INDEX=3
ucc-train   -c  configs/mask2former/mask2former-city-21k.yml \
            -o  global.find_lr=${FIND_LR} \
                global.wandb=${WANDB} \
                global.project_name=${PROJECT} \
                global.name=fold_${FOLD_INDEX}-mask2former-city_21k-large \
                data.args.train.fold_index=${FOLD_INDEX} \
                data.args.val.fold_index=${FOLD_INDEX} \
                model.args.freeze_epochs=-1 \
                data.args.SIZE=380 \
                data.args.train.loader.batch_size=8 \
                data.args.val.loader.batch_size=8 \
                trainer.lr_scheduler.name=Linear \
                trainer.optimizer.name=AdamW \
                trainer.learning_rate=1e-4 \
                trainer.optimizer.args.weight_decay=1e-2 \
                loss.smooth_factor=0.3 \
                global.SEED=3407