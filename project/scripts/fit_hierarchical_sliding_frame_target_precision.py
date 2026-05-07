from pathlib import Path

from cmdstanpy import CmdStanModel

from stan_data import make_stan_data, save_json, summarize_draws


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "hierarchical_sliding_frame_target_precision"
DISPLAY_NAME = "hierarchical_sliding_frame_target_precision"
STAN_FILE = ROOT / "models" / "hierarchical_sliding_frame_target_precision.stan"
OUT_DIR = ROOT / "outputs" / MODEL_NAME

MAX_SUBJECTS = None
FRAME_RADIUS = 1
CHAINS = 8
PARALLEL_CHAINS = 8
WARMUP = 500
SAMPLES = 500
SEED = 4210
DIAGNOSE = True

STEMS = [
    "p_target_condition",
    "p_swap_condition",
    "p_guess_condition",
    "p_target_typical",
    "p_swap_typical",
    "p_guess_typical",
    "target_precision_condition",
    "target_precision_typical",
]


data, lookup = make_stan_data(
    max_subjects=MAX_SUBJECTS,
    frame_radius=FRAME_RADIUS,
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_json(OUT_DIR / f"{MODEL_NAME}_data.json", data)
save_json(OUT_DIR / f"{MODEL_NAME}_lookup.json", lookup)

print(f"Fitting {DISPLAY_NAME} ({MODEL_NAME})")
print(f"Data: {data['N']} trials, {data['S']} subjects")
print(f"Frame radius: {FRAME_RADIUS}")
print(f"Stan file: {STAN_FILE}")
print(f"Output folder: {OUT_DIR}")

try:
    exe_file = STAN_FILE.with_suffix(".exe")
    if exe_file.exists():
        model = CmdStanModel(stan_file=str(STAN_FILE), exe_file=str(exe_file))
    else:
        model = CmdStanModel(stan_file=str(STAN_FILE))
except Exception as exc:
    print("Could not load or compile the Stan model.")
    print("The JSON data files were still written.")
    print(exc)
    raise SystemExit(2)

fit = model.sample(
    data=data,
    seed=SEED,
    chains=CHAINS,
    parallel_chains=PARALLEL_CHAINS,
    iter_warmup=WARMUP,
    iter_sampling=SAMPLES,
    output_dir=str(OUT_DIR),
    refresh=10,
)

fit.summary().to_csv(OUT_DIR / "cmdstan_summary.csv")
summary = summarize_draws(
    fit.draws_pd(),
    lookup,
    STEMS,
    OUT_DIR / "posterior_condition_summary.csv",
)

if DIAGNOSE:
    (OUT_DIR / "cmdstan_diagnose.txt").write_text(fit.diagnose(), encoding="utf-8")

print(summary.to_string(index=False))
print(f"Done: {OUT_DIR}")
