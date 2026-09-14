# The two runs that make the graph. Run this AFTER the production retrain.
#
# Both arms use an identical 400k budget so they share a learning-rate
# schedule -- comparing a 400k arm against a slice of a 3M arm is not a fair
# comparison, because the 3M arm is only a third of the way through its decay.
#
# ~40-70 min per arm.

$py = ".\venv\Scripts\python.exe"

$common = @(
  "--timesteps", "400000", "--n-envs", "8",
  "--grid-size", "12", "--grid-min", "6",
  "--max-boxes", "30", "--boxes-min", "4",
  "--randomize", "--box-source", "perfect", "--split-variety", "0.7"
)

foreach ($arm in @(
    @{ name = "graph_ref";  extra = @() },
    @{ name = "graph_flat"; extra = @("--policy", "flat") })) {
  Write-Host ""
  Write-Host "=== $($arm.name) ===" -ForegroundColor Cyan
  & $py backend\train.py @common @($arm.extra) --run-name $arm.name 2>&1 |
      Tee-Object -FilePath "$($arm.name).log"
}

Write-Host ""
Write-Host "now make the graph:" -ForegroundColor Green
Write-Host '  .\venv\Scripts\python.exe backend\plot_ablations.py --out assets\ablations.png `'
Write-Host '      --runs "our policy=tb_logs/graph_ref_1" "standard MLP=tb_logs/graph_flat_1"'
