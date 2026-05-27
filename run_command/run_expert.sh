#!/bin/bash

set -euo pipefail

# Run from the repository root even if this script is called from elsewhere.
cd "$(dirname "$0")"

TIMESTEPS="${TIMESTEPS:-300000}"
N_ENVS="${N_ENVS:-4}"
N_ROLLOUT_STEPS="${N_ROLLOUT_STEPS:-2048}"
GENERAL_DOMAIN="${GENERAL_DOMAIN:-EnergySmall_A}"
DOCKER_IMAGE="${DOCKER_IMAGE:-mipn:dev}"
USE_DOCKER="${USE_DOCKER:-1}"

run_train() {
  if [[ "$USE_DOCKER" == "1" ]]; then
    docker run --rm -v "$PWD":/app -w /app "$DOCKER_IMAGE" python3 ./train.py "$@"
  else
    python3 ./train.py "$@"
  fi
}

# 1. Expert models
expert_issues=(
  Laptop
  ItexvsCypress
  IS_BT_Acquisition
  Grocery
  thompson
  Car
  EnergySmall_A
)

expert_agent_sets=(
  # "Boulware Linear"
  # "Conceder Linear"
  # "Boulware Atlas3"
  # "Conceder Atlas3"
  # "Linear Atlas3"
  # "Atlas3 Atlas3"
  # "Boulware Boulware"
  # "Conceder Conceder" 
  # "Linear Linear" 
  "Boulware Conceder"
)

for agents in "${expert_agent_sets[@]}"; do
  read -r -a agent_args <<< "$agents"
  for issue in "${expert_issues[@]}"; do
    echo "Running expert: -a $agents -i $issue -t $TIMESTEPS -n $N_ENVS"
    run_train \
      -a "${agent_args[@]}" \
      -i "$issue" \
      --skip_venas \
      -t "$TIMESTEPS" \
      -n "$N_ENVS" \
      -rs "$N_ROLLOUT_STEPS"
  done
done

# # 2. General model
# general_issues=(
#   Laptop
#   ItexvsCypress
#   IS_BT_Acquisition
#   Grocery
#   thompson
#   Car
#   EnergySmall_A
# )

# general_agents=(
#   Boulware
#   Conceder
#   Linear
#   Atlas3
# )

# echo "Running general: -a ${general_agents[*]} -i ${general_issues[*]} -t $TIMESTEPS -n $N_ENVS -gd $GENERAL_DOMAIN"
# run_train \
#   -a "${general_agents[@]}" \
#   -i "${general_issues[@]}" \
#   --skip_venas \
#   -t "$TIMESTEPS" \
#   -n "$N_ENVS" \
#   -rs "$N_ROLLOUT_STEPS" \
#   -gd "$GENERAL_DOMAIN"
