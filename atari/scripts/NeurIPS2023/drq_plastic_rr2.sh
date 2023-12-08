cd ../..

python run_parallel.py \
    --group_name plastic \
    --exp_name drq_plastic_rr2 \
    --config_name drq \
    --seeds 0 1 2 3 4 \
    --num_games 26 \
    --num_devices 3 \
    --num_exp_per_device 3 \
    --overrides agent.optimizer.type=sam \
                agent.optimizer.base_optimizer=Adam \
                agent.optimizer.rho=0.1 \
                agent.optimizer.adaptive=False \
                agent.optimizer.sam_head=False \
                agent.optimize_per_env_step=2 \
                agent.reset_type=llf \
                agent.reset_target=True \
                agent.reset_per_optimize_step=40000 \
                model.policy.activation=CReLU_output \
                model.backbone.normalization=layernorm \
                agent.shrink_perturb=False 

