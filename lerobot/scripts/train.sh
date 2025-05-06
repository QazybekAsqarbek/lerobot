# train diff policy on pusht - works
# python lerobot/scripts/train.py \
#     --dataset.repo_id=lerobot/pusht \
#     --policy.type=diffusion \
#     --env.type=pusht


# # train diff policy on agibot - ?
# python lerobot/scripts/train.py \
#     --dataset.repo_id=lerobot/agibot \
#     --policy.type=diffusion \
#     --env.type=agibot



# # CLEARML_CONFIG_FILE=.clearml/clearml.conf DATA_DIR=../dataset_collection/data/preprocessed_kf hydra.job.name="experiment_2" python lerobot/scripts/train.py env=cobot_real env.fps=1 policy=diffusion_actor_cobot dataset_repo_id=3d_manipulation/real_world_kf_table_setting_first_version



# stages:
# 1. upload agibot dataset to hub
# 2. train diff policy on agibot


# ===== Stage 1 =====
# export CAMERA_INDICES="1,2,3"
# export KEYFRAMES_ONLY=0
# export INTERPOLATION_LENGTH=48
# export INCLUDE_LAST_KEYFRAME=1
# export MIN_NUM_FRAMES=15
# export DATASET_FPS=15
# export INSTRUCTIONS_PATH="/home/jovyan/kulikov/3d_diffuser_actor/configs/instructions_q3_benchmark_all_random.json"
# export REQUIRE_KEYFRAMES_CONSISTENCY_ACROSS_VARIATION=1
# python lerobot/scripts/push_dataset_to_hub.py \
# --raw-dir ../dataset_collection/data/raw/table_setting_first_version \
# --raw-format cobot_zarr \
# --repo-id 3d_manipulation/real_world_table_setting_first_version \
# --local-dir ../dataset_collection/data/preprocessed/3d_manipulation/real_world_table_setting_first_version \
# --push-to-hub 0 \
# --fps 15 \
# --video 1 \
# --force-override 1
# # ===== Stage 1 =====


# ===== Train on custom dataset created by scripts/convert_to_lerobot.py =====
# This section shows how to train a policy on a dataset you created using
# the 'scripts/convert_to_lerobot.py' script.

# --- Prerequisites ---
# 1. You must have already run 'scripts/convert_to_lerobot.py' to generate your dataset.
#    Example command for conversion:
#    python scripts/convert_to_lerobot.py \\
#        --data_dir /path/to/your/raw/robot/data \\
#        --save_dir /path/to/your/converted_lerobot_datasets \\
#        --repo_id my_robot_tasks/dataset_v1 \\
#        --preprocess_video # Optional

# --- Setup for Training ---
# 1. Set the LEROBOT_HOME environment variable.
#    This should be the '--save_dir' you used when running 'scripts/convert_to_lerobot.py'.
#    LeRobot will look for datasets inside this directory.
#    Example: If --save_dir was /srv/data/my_converted_datasets
#             Then: export LEROBOT_HOME="/srv/data/my_converted_datasets"
export HF_LEROBOT_HOME=/home/jovyan/shares/SR006.nfs2/askarbek/data/preprocessed/

# 2. Set MY_DATASET_REPO_ID to the '--repo_id' you used with 'scripts/convert_to_lerobot.py'.
#    This is the name of your dataset directory within LEROBOT_HOME.
#    Example: If --repo_id was my_robot_tasks/dataset_v1
#             Then: export MY_DATASET_REPO_ID="my_robot_tasks/dataset_v1"
export MY_DATASET_REPO_ID=agibot/agibotdigital

# --- Run Training ---
# Notes:
#   - '--dataset.repo_id': Should match MY_DATASET_REPO_ID.
#   - '--policy.type': Choose your desired policy (e.g., diffusion, ddpm).
#   - '--env.type': Should match the 'robot_type' from convert_to_lerobot.py (default "a2d").
#   - '--env.fps': Should match the 'fps' from convert_to_lerobot.py (default 30).
#   - '--hydra.run.dir': Specifies the output directory for logs, checkpoints, etc.
#     The example below saves outputs within a subfolder of LEROBOT_HOME.
#
# Before running, replace the placeholder values for LEROBOT_HOME and MY_DATASET_REPO_ID above.

echo "Starting LeRobot training..."
echo "  LEROBOT_HOME (Dataset Root): $HF_LEROBOT_HOME"
echo "  Dataset Repo ID (Subdirectory): $MY_DATASET_REPO_ID"
echo "  Policy Type: diffusion"
echo "  Environment Type: a2d"
echo "  Environment FPS: 30"

# Ensure LEROBOT_HOME and MY_DATASET_REPO_ID are not placeholders
if [ "$HF_LEROBOT_HOME" = "<YOUR_CONVERT_SCRIPT_SAVE_DIR_HERE>" ] || [ "$MY_DATASET_REPO_ID" = "<YOUR_CONVERT_SCRIPT_REPO_ID_HERE>" ]; then
    echo "ERROR: Please replace placeholder values for LEROBOT_HOME and MY_DATASET_REPO_ID in the script."
    exit 1
fi
export WANDB_API_KEY=9f478cbfd843b844545c0c9efdf3c72cb34ce9a1

export CUDA_VISIBLE_DEVICES=0

# Train 1 Vanilla Diffusion Policy

# python lerobot/scripts/train.py \
#     --dataset.repo_id=$MY_DATASET_REPO_ID \
#     --policy.type=diffusion \
#     --wandb.project="agibotdigital" \
#     --wandb.entity="kazdovakin" \
#     --wandb.enable=true \
#     --batch_size=64 \
#     --num_workers=12 \
#     --policy.use_amp=true \
#     --dataset.video_backend=torchcodec \
#     --dataset.use_imagenet_stats=false \
#     --seed=42 \
#     --steps=1000000


# Train 2 Diffusion Policy with VGGT Encoder
ENCODER_TYPE=vggt
FREEZE_ENCODER=true
export CUDA_VISIBLE_DEVICES=1

python lerobot/scripts/train.py \
    --dataset.repo_id=$MY_DATASET_REPO_ID \
    --policy.type=diffusion \
    --policy.encoder_type=${ENCODER_TYPE:-vggt} \
    --policy.freeze_encoder=${FREEZE_ENCODER:-false} \
    --wandb.project="agibotdigital" \
    --wandb.entity="kazdovakin" \
    --wandb.enable=true \
    --batch_size=64 \
    --num_workers=12 \
    --policy.use_amp=true \
    --dataset.video_backend=torchcodec \
    --dataset.use_imagenet_stats=false \
    --seed=42 \
    --steps=1000000


    # For ClearML logging (ensure clearml is installed and configured):
    # CLEARML_CONFIG_FILE=~/.clearml.conf python lerobot/scripts/train.py ... (add CLEARML_CONFIG_FILE before python)
    # train.add_clearml_logger=True \\
    # train.clearml_project_name="LeRobot Custom Training" \\
    # train.clearml_task_name="Training run - \${dataset.repo_id} - \${policy.type}" \\

echo "Training script finished."
# ===== End of custom dataset training section =====

# Usage:
# usage: train.py [-h] [--config_path str] [--dataset str] [--dataset.repo_id str] [--dataset.root str] [--dataset.episodes str] [--image_transforms str]
#                 [--dataset.image_transforms.enable str] [--dataset.image_transforms.max_num_transforms str] [--dataset.image_transforms.random_order str]
#                 [--dataset.image_transforms.tfs str] [--dataset.revision str] [--dataset.use_imagenet_stats str] [--dataset.video_backend str] [--env str]
#                 [--env.type {aloha,pusht,xarm,a2d}] [--env.visualization_width str] [--env.visualization_height str] [--env.task str] [--env.fps str]
#                 [--env.features str] [--env.features_map str] [--env.episode_length str] [--env.obs_type str] [--env.render_mode str] [--policy str]
#                 [--policy.type {act,diffusion,pi0,tdmpc,vqbet,pi0fast}] [--policy.replace_final_stride_with_dilation str] [--policy.pre_norm str]
#                 [--policy.dim_model str] [--policy.n_heads str] [--policy.dim_feedforward str] [--policy.feedforward_activation str]
#                 [--policy.n_encoder_layers str] [--policy.n_decoder_layers str] [--policy.use_vae str] [--policy.n_vae_encoder_layers str]
#                 [--policy.temporal_ensemble_coeff str] [--policy.kl_weight str] [--policy.optimizer_lr_backbone str] [--policy.drop_n_last_frames str]
#                 [--policy.use_separate_rgb_encoder_per_camera str] [--policy.down_dims str] [--policy.kernel_size str] [--policy.n_groups str]
#                 [--policy.diffusion_step_embed_dim str] [--policy.use_film_scale_modulation str] [--policy.noise_scheduler_type str]
#                 [--policy.num_train_timesteps str] [--policy.beta_schedule str] [--policy.beta_start str] [--policy.beta_end str]
#                 [--policy.prediction_type str] [--policy.clip_sample str] [--policy.clip_sample_range str] [--policy.num_inference_steps str]
#                 [--policy.do_mask_loss_for_padding str] [--policy.scheduler_name str] [--policy.num_steps str] [--policy.attention_implementation str]
#                 [--policy.train_expert_only str] [--policy.train_state_proj str] [--policy.n_action_repeats str] [--policy.horizon str]
#                 [--policy.image_encoder_hidden_dim str] [--policy.state_encoder_hidden_dim str] [--policy.latent_dim str] [--policy.q_ensemble_size str]
#                 [--policy.mlp_dim str] [--policy.discount str] [--policy.use_mpc str] [--policy.cem_iterations str] [--policy.max_std str]
#                 [--policy.min_std str] [--policy.n_gaussian_samples str] [--policy.n_pi_samples str] [--policy.uncertainty_regularizer_coeff str]
#                 [--policy.n_elites str] [--policy.elite_weighting_temperature str] [--policy.gaussian_mean_momentum str]
#                 [--policy.max_random_shift_ratio str] [--policy.reward_coeff str] [--policy.expectile_weight str] [--policy.value_coeff str]
#                 [--policy.consistency_coeff str] [--policy.advantage_scaling str] [--policy.pi_coeff str] [--policy.temporal_decay_coeff str]
#                 [--policy.target_model_momentum str] [--policy.n_action_pred_token str] [--policy.action_chunk_size str] [--policy.vision_backbone str]
#                 [--policy.crop_shape str] [--policy.crop_is_random str] [--policy.pretrained_backbone_weights str] [--policy.use_group_norm str]
#                 [--policy.spatial_softmax_num_keypoints str] [--policy.n_vqvae_training_steps str] [--policy.vqvae_n_embed str]
#                 [--policy.vqvae_embedding_dim str] [--policy.vqvae_enc_hidden_dim str] [--policy.gpt_block_size str] [--policy.gpt_input_dim str]
#                 [--policy.gpt_output_dim str] [--policy.gpt_n_layer str] [--policy.gpt_n_head str] [--policy.gpt_hidden_dim str] [--policy.dropout str]
#                 [--policy.mlp_hidden_dim str] [--policy.offset_loss_weight str] [--policy.primary_code_loss_weight str]
#                 [--policy.secondary_code_loss_weight str] [--policy.bet_softmax_temperature str] [--policy.sequentially_select str]
#                 [--policy.optimizer_vqvae_lr str] [--policy.optimizer_vqvae_weight_decay str] [--policy.n_obs_steps str] [--policy.normalization_mapping str]
#                 [--policy.input_features str] [--policy.output_features str] [--policy.device str] [--policy.use_amp str] [--policy.chunk_size str]
#                 [--policy.n_action_steps str] [--policy.max_state_dim str] [--policy.max_action_dim str] [--policy.resize_imgs_with_padding str]
#                 [--policy.interpolate_like_pi str] [--policy.empty_cameras str] [--policy.adapt_to_pi_aloha str] [--policy.use_delta_joint_actions_aloha str]
#                 [--policy.tokenizer_max_length str] [--policy.proj_width str] [--policy.max_decoding_steps str] [--policy.fast_skip_tokens str]
#                 [--policy.max_input_seq_len str] [--policy.use_cache str] [--policy.freeze_vision_encoder str] [--policy.freeze_lm_head str]
#                 [--policy.optimizer_lr str] [--policy.optimizer_betas str] [--policy.optimizer_eps str] [--policy.optimizer_weight_decay str]
#                 [--policy.scheduler_warmup_steps str] [--policy.scheduler_decay_steps str] [--policy.scheduler_decay_lr str] [--policy.checkpoint_path str]
#                 [--policy.padding_side str] [--policy.precision str] [--policy.grad_clip_norm str] [--policy.relaxed_action_decoding str] [--output_dir str]
#                 [--job_name str] [--resume str] [--seed str] [--num_workers str] [--batch_size str] [--steps str] [--eval_freq str] [--log_freq str]
#                 [--save_checkpoint str] [--save_freq str] [--use_policy_training_preset str] [--optimizer str] [--optimizer.type {adam,adamw,sgd}]
#                 [--optimizer.betas str] [--optimizer.eps str] [--optimizer.lr str] [--optimizer.weight_decay str] [--optimizer.grad_clip_norm str]
#                 [--optimizer.momentum str] [--optimizer.dampening str] [--optimizer.nesterov str] [--scheduler str]
#                 [--scheduler.type {diffuser,vqbet,cosine_decay_with_warmup}] [--scheduler.name str] [--scheduler.num_vqvae_training_steps str]
#                 [--scheduler.num_cycles str] [--scheduler.num_warmup_steps str] [--scheduler.num_decay_steps str] [--scheduler.peak_lr str]
#                 [--scheduler.decay_lr str] [--eval str] [--eval.n_episodes str] [--eval.batch_size str] [--eval.use_async_envs str] [--wandb str]
#                 [--wandb.enable str] [--wandb.disable_artifact str] [--wandb.project str] [--wandb.entity str] [--wandb.notes str] [--wandb.run_id str]
#                 [--wandb.mode str]
