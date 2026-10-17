from pathlib import Path
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/clawgym-official-30b-formal-rl-local-retain-all-20260809-085700"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)
C = {"blue":"#2673B8","orange":"#E2842A","green":"#2B8C69","red":"#C94B48",
     "purple":"#7A5AA6","gray":"#69737D","gold":"#C49A22"}
SFT_BASELINE_REWARD = 0.5408851525911653

def style():
    plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"#FBFCFE",
        "axes.edgecolor":"#C7CED6","axes.titleweight":"bold","axes.titlesize":12,
        "axes.labelsize":10,"font.size":9.5,"legend.fontsize":8.5,
        "lines.linewidth":2.0,"savefig.facecolor":"white"})

def load_scalars():
    merged = {}
    for path in glob.glob(str(RUN / "tensorboard" / "events.out.tfevents.*")):
        event = EventAccumulator(path, size_guidance={"scalars":0})
        event.Reload()
        for tag in event.Tags().get("scalars", []):
            for item in event.Scalars(tag):
                key = (tag, item.step)
                old = merged.get(key)
                if old is None or item.wall_time >= old[0]:
                    merged[key] = (item.wall_time, item.value)
    result = {}
    for tag in sorted({tag for tag, _ in merged}):
        pts = sorted((step, value) for (ptag, step), (_, value) in merged.items() if ptag == tag)
        result[tag] = ([p[0] for p in pts], [p[1] for p in pts])
    return result

def series(data, tag):
    return data.get(tag, ([], []))

def ma(values, window=5):
    return [sum(values[max(0,i-window+1):i+1])/len(values[max(0,i-window+1):i+1]) for i in range(len(values))]

def polish(ax, xlabel=None, ylabel=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.22, linewidth=.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def line(ax, data, tag, label, color, marker="o", percent=False):
    x, y = series(data, tag)
    values = [100*v for v in y] if percent else y
    if x: ax.plot(x, values, marker=marker, markersize=3.5, label=label, color=color)
    return x, values

def latest_label(ax, x, y, color):
    if not x: return
    ax.scatter([x[-1]],[y[-1]],s=55,color=color,zorder=5)
    ax.annotate(f"latest {y[-1]:.3f}",(x[-1],y[-1]),xytext=(-54,15),
        textcoords="offset points",color=color,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25",fc="white",ec=color,alpha=.92))

def latest_sci_label(ax, x, y, color, scale=1.0, suffix=""):
    if not x:
        return
    value = y[-1] * scale
    ax.scatter([x[-1]], [value], s=55, color=color, zorder=5)
    ax.annotate(
        f"step {x[-1]}: {value:.3f}{suffix}",
        (x[-1], value),
        xytext=(-92, 15),
        textcoords="offset points",
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=.92),
    )

def save_reward(data):
    fig, axes = plt.subplots(2,1,figsize=(13.2,8.0),sharex=True)
    fig.suptitle("Formal RL reward — two different aggregation levels",fontsize=16,fontweight="bold",y=.985)
    ax = axes[0]
    x,y = line(ax,data,"openclaw/avg_reward_submitted","64-trajectory mean",C["blue"])
    if x: ax.plot(x,ma(y),color=C["orange"],linewidth=2.5,label="5-rollout moving mean")
    latest_label(ax,x,y,C["blue"])
    ex,ey = series(data,"eval/clawgym_eval-avg_reward")
    if ex: ax.scatter(ex,ey,marker="D",s=70,color=C["green"],label="200-task RL eval")
    ax.axhline(
        SFT_BASELINE_REWARD,
        color=C["gray"],
        linestyle="--",
        linewidth=1.4,
        label="SFT 200-task reference (0.5409)",
    )
    ax.set_title("A. Environment-level reward (one vote per logical trajectory)",loc="left")
    ax.set_ylim(0,1.02); polish(ax,ylabel="Reward")
    ax.legend(loc="lower left",ncol=2,frameon=True,framealpha=.9)

    ax = axes[1]
    x,y = line(ax,data,"rollout/raw_reward","Training-sample weighted reward",C["purple"],marker="s")
    if x: ax.plot(x,ma(y),color=C["gold"],linewidth=2.5,label="5-step moving mean")
    latest_label(ax,x,y,C["purple"])
    ax.set_title("B. Optimizer input reward (long, segmented trajectories receive more sample weight)",loc="left")
    ax.set_ylim(0,1.02); polish(ax,xlabel="Rollout / optimizer step",ylabel="Reward")
    ax.legend(loc="lower left",ncol=2,frameon=True,framealpha=.9)
    fig.text(.5,.012,"Panels are separated because their denominators differ; do not compare them point-for-point.",
        ha="center",color=C["gray"],fontsize=9)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.09,hspace=.28)
    fig.savefig(OUT/"reward_curves.png",dpi=200); plt.close(fig)

def save_stability(data):
    fig,axes = plt.subplots(2,2,figsize=(14,9))
    fig.suptitle("Optimizer stability diagnostics",fontsize=16,fontweight="bold",y=.985)
    ax=axes[0,0]
    line(ax,data,"train/pre_clip_grad_norm","Pre-clip",C["red"])
    line(ax,data,"train/post_clip_grad_norm","Post-clip",C["green"],marker="s")
    ax.axhline(1,color=C["gray"],linestyle="--",linewidth=1.2,label="clip-grad = 1.0")
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_title("A. Gradient norm (log scale)",loc="left"); polish(ax,ylabel="L2 norm")
    ax.legend(loc="upper left",frameon=True,framealpha=.9)
    ax=axes[0,1]
    line(ax,data,"train/kl_loss","Low-variance reference KL",C["blue"])
    ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,-3))
    ax.set_title("B. KL proxy to frozen initial SFT",loc="left")
    polish(ax,ylabel="Approximate KL")
    ax=axes[1,0]
    line(ax,data,"train/pg_clipfrac","Clipped response tokens",C["purple"],percent=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=2))
    ax.set_title("C. PPO policy-ratio clipping",loc="left")
    polish(ax,xlabel="Optimizer step",ylabel="Token fraction")
    ax=axes[1,1]
    line(ax,data,"train/train_rollout_logprob_abs_diff","|Megatron − SGLang logprob|",C["orange"])
    ax.set_title("D. Train / rollout logprob numerical gap",loc="left")
    polish(ax,xlabel="Optimizer step",ylabel="Mean absolute difference")
    fig.text(.5,.012,"The log axis preserves isolated pre-clip spikes; post-clip values verify updates were capped at 1.0.",
        ha="center",color=C["gray"],fontsize=9)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.08,wspace=.22,hspace=.30)
    fig.savefig(OUT/"optimizer_stability.png",dpi=200); plt.close(fig)

def save_kl(data):
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.2), sharex=True)
    fig.suptitle(
        "KL diagnostics — reference drift and PPO update are different quantities",
        fontsize=16, fontweight="bold", y=.985,
    )

    ax = axes[0]
    x, y = series(data, "train/kl_loss")
    scaled = [1000 * value for value in y]
    if x:
        ax.plot(x, scaled, marker="o", markersize=3.5, color=C["blue"],
                label="Per-step low-variance KL proxy")
        ax.plot(x, ma(scaled), color=C["orange"], linewidth=2.5,
                label="5-step moving mean")
    latest_sci_label(ax, x, y, C["blue"], scale=1000, suffix=" ×10⁻³")
    ax.axhline(0, color=C["gray"], linewidth=1)
    ax.set_title("A. Current policy vs frozen ClawGym-30B-A3B SFT reference", loc="left")
    polish(ax, ylabel="Approximate KL (×10⁻³)")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=.9)
    ax.text(.995, .06,
        "batch-local response-token estimate · low_var_kl · coefficient = 0",
        transform=ax.transAxes, ha="right", va="bottom", color=C["gray"], fontsize=8.5)

    ax = axes[1]
    x, y = series(data, "train/ppo_kl")
    scaled = [10000 * value for value in y]
    if x:
        ax.plot(x, scaled, marker="s", markersize=3.5, color=C["purple"],
                label="Signed PPO old/current estimator")
        ax.plot(x, ma(scaled), color=C["gold"], linewidth=2.5,
                label="5-step moving mean")
    latest_sci_label(ax, x, y, C["purple"], scale=10000, suffix=" ×10⁻⁴")
    ax.axhline(0, color=C["gray"], linewidth=1.1)
    ax.set_title("B. PPO behavior-policy vs current-policy update diagnostic (signed)", loc="left")
    polish(ax, xlabel="Optimizer step", ylabel="Signed estimate (×10⁻⁴)")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=.9)
    ax.text(.995, .06,
        "train/ppo_kl · may be negative from finite-sample estimation · not KL to SFT",
        transform=ax.transAxes, ha="right", va="bottom", color=C["gray"], fontsize=8.5)

    fig.text(.5, .012,
        "Panel A measures policy drift from the frozen SFT reference; Panel B only diagnoses each PPO update.",
        ha="center", color=C["gray"], fontsize=9)
    fig.subplots_adjust(left=.085, right=.975, top=.91, bottom=.09, hspace=.30)
    fig.savefig(OUT/"kl_diagnostics.png", dpi=200)
    plt.close(fig)

def save_context(data):
    fig,axes=plt.subplots(2,2,figsize=(14,9.2))
    fig.suptitle("Trajectory health and context-rewrite diagnostics",fontsize=16,fontweight="bold",y=.985)
    ax=axes[0,0]
    line(ax,data,"openclaw/avg_samples_per_trajectory","Training samples / trajectory",C["blue"])
    ax.axhline(1,color=C["gray"],linestyle="--",linewidth=1.1,label="No segmentation")
    ax.set_title("A. Segment amplification",loc="left"); polish(ax,ylabel="Samples per trajectory")
    ax.legend(loc="upper left",frameon=True,framealpha=.9)
    ax=axes[0,1]
    line(ax,data,"openclaw/tool_result_pruning_events_total","Pruning events",C["red"])
    ax2=ax.twinx()
    line(ax2,data,"openclaw/session_compactions_total","Session compactions",C["green"],marker="s")
    ax.set_title("B. History rewrite events per 64 trajectories",loc="left"); polish(ax,ylabel="Event count")
    ax2.set_ylabel("Session compactions",color=C["green"])
    ax2.set_ylim(-.05,1.2)
    ax2.spines["top"].set_visible(False)
    h1,l1=ax.get_legend_handles_labels()
    h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,loc="upper left",frameon=True,framealpha=.9)
    ax=axes[1,0]
    line(ax,data,"rollout/trajectories_hit_response_limit","Hit 8,192-token turn limit",C["orange"])
    line(ax,data,"rollout/repetition_degenerated_trajectories","Repetition degeneration",C["purple"],marker="s")
    line(ax,data,"rollout/termination/max_agent_steps","Hit 96 agent steps",C["gray"],marker="^")
    ax.set_title("C. Policy-side long-tail failures",loc="left")
    polish(ax,xlabel="Rollout step",ylabel="Trajectories / 64")
    ax.legend(loc="upper left",frameon=True,framealpha=.9)
    ax=axes[1,1]
    line(ax,data,"openclaw/avg_task_duration_sec","Mean",C["blue"])
    line(ax,data,"openclaw/p90_task_duration_sec","P90",C["gold"],marker="s")
    line(ax,data,"openclaw/max_task_duration_sec","Maximum",C["red"],marker="^")
    ax.axhline(1200,color=C["gray"],linestyle="--",linewidth=1.1,label="Outer timeout = 1,200 s")
    ax.set_title("D. End-to-end trajectory duration",loc="left")
    polish(ax,xlabel="Rollout step",ylabel="Seconds")
    ax.legend(loc="upper left",ncol=2,frameon=True,framealpha=.9)
    fig.text(.5,.012,"Three context overflows occurred (rollouts 12, 25 and 53); unclassified history rewrites remain zero.",
        ha="center",color=C["gray"],fontsize=9)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.08,wspace=.22,hspace=.30)
    fig.savefig(OUT/"context_segmentation.png",dpi=200); plt.close(fig)

if __name__=="__main__":
    style(); data=load_scalars()
    save_reward(data); save_stability(data); save_kl(data); save_context(data)
