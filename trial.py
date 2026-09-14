import os, re, json, sys, yaml
from pathlib import Path
LIBERO_PLUS_PATH = Path(__file__).parent.parent.parent / "third_party" / "libero_plus"
if str(LIBERO_PLUS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBERO_PLUS_PATH))

# Pin LIBERO's path config to this repo's libero_plus tree. Without this, LIBERO reads
# the machine-global ~/.libero/config.yaml, which is written once by whichever LIBERO
# checkout was imported first and may point at an unrelated one that lacks the
# LIBERO-plus perturbation bddl_files/init_files.
_LIBERO_CONFIG_DIR = LIBERO_PLUS_PATH / ".libero"
_LIBERO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_CONFIG_DIR)
_LIBERO_ROOT = LIBERO_PLUS_PATH / "libero" / "libero"
_LIBERO_CONFIG_FILE = _LIBERO_CONFIG_DIR / "config.yaml"
if not _LIBERO_CONFIG_FILE.exists():
    with _LIBERO_CONFIG_FILE.open("w") as _f:
        yaml.safe_dump(
            {
                "benchmark_root": str(_LIBERO_ROOT),
                "bddl_files": str(_LIBERO_ROOT / "bddl_files"),
                "init_states": str(_LIBERO_ROOT / "init_files"),
                "datasets": str(_LIBERO_ROOT.parent / "datasets"),
                "assets": str(_LIBERO_ROOT / "assets"),
            },
            _f,
        )
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import cv2


def save_obs(obs, step):
    """MuJoCo renders bottom-up and cv2 wants BGR: flip rows, reverse channels."""
    img = obs[agentview]
    cv2.imwrite(os.path.join( f"{step:04d}_{agentview}.png"), img[::-1, :, ::-1])

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()   # keep task_order_index=0

# --- pick a Robot Initial States task -------------------------------------
cls = os.path.join(get_libero_path("benchmark_root"), "benchmark", "task_classification.json")
rows = json.load(open(cls))[task_suite_name]
robot_tasks = [i for i, r in enumerate(rows) if r["category"] == "Robot Initial States"]
task_id = robot_tasks[0]        # 289 for libero_10; 393 available
# --------------------------------------------------------------------------

task_id = 2113
task = task_suite.get_task(task_id)
task_description = re.split(r"\s+view\s+-?\d", task.language)[0]   # see note below
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=128, camera_widths=128)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])       # note: 1 trial per task, not 50

dummy_action = [0.] * 7
for step in range(2):                   # these steps are what pull the arm to the perturbed pose
    obs, reward, done, info = env.step(dummy_action)
    # Cache the obs as png file
    save_obs(obs, step)

env.close()
