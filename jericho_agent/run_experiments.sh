#!/bin/bash

# ================= Configuration =================
# 1. Define the module path (do not include the `.py` suffix).
# MODULE="jericho_agent.evaluate"

# # 2. Define the games to run (space-separated).
# GAMES="detective zork1 temple"
# # GAMES="detective"

# # 3. Shared experiment settings.
# SEEDS=1
# REPEATS=15
# PROMPT_TAG_DEFAULT="DEFAULT"
# PROMPT_TAG_OPT2="OPT_V2"
# STEPS=30


# echo ""
# echo "▶️  [P2 Budget] Running GPT-5 (Teacher) -> Qwen3-8b (Student)..."
# # OpenRouter/LiteLLM usually expects a provider prefix such as `qwen/`.
# python -m $MODULE \
#   --meta_model "openai/gpt-5" \
#   --actor_model "qwen/qwen3-8b" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG



# echo ""
# echo "▶️  [P8 ] Running GPT5 (Teacher) -> glm-4.7-flash (Student)..."
# python -m $MODULE \
#   --meta_model "openai/gpt-5" \
#   --actor_model "z-ai/glm-4.7-flash" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG

# echo ""

# echo ""
# echo "▶️  [P9] z-ai/glm-4.7 (Teacher) -> z-ai/glm-4.7-flash (Student)..."
# python -m $MODULE \
#   --meta_model "z-ai/glm-4.7" \
#   --actor_model "z-ai/glm-4.7-flash" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG

# echo ""
# echo "▶️  [New 1] Gemini 3 Flash (Teacher) -> Gemini 3 Flash (Student)..."
# # Assume "3 flash" refers to Google Gemini 2.0 Flash.
# python -m $MODULE \
#   --meta_model "google/gemini-3-flash-preview" \
#   --actor_model "google/gemini-3-flash-preview" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG

# echo ""
# echo "▶️  [New 2] GLM-4.7 flash (Teacher) -> GLM-4.7 flash (Student)..."
# python -m $MODULE \
#   --meta_model "z-ai/glm-4.7-flash" \
#   --actor_model "z-ai/glm-4.7-flash" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG

# echo ""
# echo "▶️  [New 3] Qwen3-8b (Teacher) -> Qwen3-8b (Student)..."
# python -m $MODULE \
#   --meta_model "qwen/qwen3-8b" \
#   --actor_model "qwen/qwen3-8b" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG

# echo ""
# echo "▶️  [New 4] GPT-5 (Teacher) -> GPT-OSS-120b (Student)..."
# python -m $MODULE \
#   --meta_model "openai/gpt-5" \
#   --actor_model "openai/gpt-oss-120b" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG \
#   --disable_actor_thinking "False"

# echo ""
# echo "▶️  [New 5] GPT-5 (Teacher) -> qwen3-coder-next..."
# python -m $MODULE \
#   --meta_model "openai/gpt-5" \
#   --actor_model "qwen/qwen3-coder-next" \
#   --games $GAMES \
#   --seeds $SEEDS \
#   --repeats $REPEATS \
#   --env_step_limit $STEPS \
#   --prompt_tag $PROMPT_TAG \
#   --max_episodes 3

# ==========================================
# GPT-5 Teacher Group
# ==========================================

# detective zork1 temple - 5 eps
# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 5 eps | opt13 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 5 eps | opt9 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_9 \
#     --max_episodes 5 \
#     --env_step_limit 50

# # detective zork1 temple - 10 eps
# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 10 eps | default +5"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag DEFAULT \
#     --max_episodes 10 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 10 eps | opt13 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_13 \
#     --max_episodes 10 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 10 eps | opt9 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_9 \
#     --max_episodes 10 \
#     --env_step_limit 50

# # balances library zork3(OOD) - 5 eps
# echo ""
# echo "▶️ GPT-5 Teacher | balances library zork3 | 5 eps | default +15"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag DEFAULT \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | balances library zork3 | 5 eps | opt13 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | balances library zork3 | 5 eps | opt9 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_9 \
#     --max_episodes 5 \
#     --env_step_limit 50

# # # balances library zork3(OOD) - 10 eps
# # echo ""
# # echo "▶️ GPT-5 Teacher | balances library zork3 | 10 eps | default +5"
# # python -m jericho_agent.evaluate \
# #     --meta_model "openai/gpt-5" \
# #     --actor_model "google/gemini-3-flash-preview" \
# #     --games balances library zork3 \
# #     --seeds 1 \
# #     --repeats 5 \
# #     --prompt_tag DEFAULT \
# #     --max_episodes 10 \
# #     --env_step_limit 50

# # echo ""
# # echo "▶️ GPT-5 Teacher | balances library zork3 | 10 eps | opt13 +5"
# # python -m jericho_agent.evaluate \
# #     --meta_model "openai/gpt-5" \
# #     --actor_model "google/gemini-3-flash-preview" \
# #     --games balances library zork3 \
# #     --seeds 1 \
# #     --repeats 5 \
# #     --prompt_tag OPT_13 \
# #     --max_episodes 10 \
# #     --env_step_limit 50

# # echo ""
# # echo "▶️ GPT-5 Teacher | balances library zork3 | 10 eps | opt9 +5"
# # python -m jericho_agent.evaluate \
# #     --meta_model "openai/gpt-5" \
# #     --actor_model "google/gemini-3-flash-preview" \
# #     --games balances library zork3 \
# #     --seeds 1 \
# #     --repeats 5 \
# #     --prompt_tag OPT_9 \
# #     --max_episodes 10 \
# #     --env_step_limit 50


# # ==========================================
# # Gemini 3 Flash Teacher Group
# # ==========================================

# # detective zork1 temple - 5 eps
# echo ""
# echo "▶️ Gemini 3 Flash Teacher | detective zork1 temple | 5 eps | opt10 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_10 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ Gemini 3 Flash Teacher | detective zork1 temple | 5 eps | opt13 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# # detective zork1 temple - 10 eps
# echo ""
# echo "▶️ Gemini 3 Flash Teacher | detective zork1 temple | 10 eps | default +5"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag DEFAULT \
#     --max_episodes 10 \
#     --env_step_limit 50

# echo ""
# echo "▶️ Gemini 3 Flash Teacher | detective zork1 temple | 10 eps | opt10 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_10 \
#     --max_episodes 10 \
#     --env_step_limit 50

# # balances library zork3(OOD) - 5 eps
# echo ""
# echo "▶️ Gemini 3 Flash Teacher | balances library zork3 | 5 eps | default +15"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag DEFAULT \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ Gemini 3 Flash Teacher | balances library zork3 | 5 eps | opt10 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_10 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ Gemini 3 Flash Teacher | balances library zork3 | 5 eps | opt13 +15"
# python -m jericho_agent.evaluate \
#     --meta_model "google/gemini-3-flash-preview" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 15 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# ==========================================
# Claude Sonnet 4.6 Teacher Group
# ==========================================

# detective zork1 temple - 5 eps
# echo ""
# echo "▶️ Claude Sonnet 4.6 Teacher | detective zork1 temple | 5 eps | opt13 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "claude-sonnet-4.6" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# # balances library zork3(OOD) - 5 eps
# echo ""
# echo "▶️ Claude Sonnet 4.6 Teacher | balances library zork3 | 5 eps | opt11 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "claude-sonnet-4.6" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_11 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ Claude Sonnet 4.6 Teacher | balances library zork3 | 5 eps | opt13 +5"
# python -m jericho_agent.evaluate \
#     --meta_model "claude-sonnet-4.6" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games balances library zork3 \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag OPT_13 \
#     --max_episodes 5 \
#     --env_step_limit 50

# echo ""
# echo "▶️ GPT-5 Teacher | detective zork1 temple | 5 eps | default +5"
# python -m jericho_agent.evaluate \
#     --meta_model "openai/gpt-5" \
#     --actor_model "google/gemini-3-flash-preview" \
#     --games detective zork1 temple \
#     --seeds 1 \
#     --repeats 5 \
#     --prompt_tag DEFAULT \
#     --max_episodes 10 \
#     --env_step_limit 50

echo "▶️ GLM 5 Teacher | detective zork1 temple | 5 eps | default +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games detective zork1 temple \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag DEFAULT \
    --max_episodes 5 \
    --env_step_limit 50

results/eval/eval_glm-5_gemini-3-flash-preview_DEFAULT_20260228_082716.pkl


echo "▶️ GLM 5 Teacher | balances library zork3 (OOD) | 5 eps | default +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games balances library zork3 \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag DEFAULT \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_DEFAULT_20260228_091113.pkl

echo "▶️ GLM 5 Teacher | detective zork1 temple | 5 eps | opt8 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games detective zork1 temple \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_8 \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_OPT_8_20260228_094956.pkl


echo "▶️ GLM 5 Teacher | balances library zork3 (OOD) | 5 eps | opt8 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games balances library zork3 \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_8 \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_OPT_8_20260228_103926.pkl

echo "▶️ GLM 5 Teacher | detective zork1 temple | 5 eps | opt14 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games detective zork1 temple \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_14 \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_OPT_14_20260228_112156.pkl

echo "▶️ GLM 5 Teacher | balances library zork3 (OOD) | 5 eps | opt14 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games balances library zork3 \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_14 \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_OPT_14_20260228_115820.pkl

echo "▶️ GLM 5 Teacher | detective zork1 temple | 5 eps | opt13 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games detective zork1 temple \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_13 \
    --max_episodes 5 \
    --env_step_limit 50
results/eval/eval_glm-5_gemini-3-flash-preview_OPT_13_20260228_123357.pkl



echo "▶️ GLM 5 Teacher | balances library zork3 (OOD) | 5 eps | opt13 +15"
python -m jericho_agent.evaluate \
    --meta_model "z-ai/glm-5" \
    --actor_model "google/gemini-3-flash-preview" \
    --games balances library zork3 \
    --seeds 1 \
    --repeats 15 \
    --prompt_tag OPT_13 \
    --max_episodes 5 \
    --env_step_limit 50
echo "✅ All experiments completed! Check results/eval/ and results/plots/."
