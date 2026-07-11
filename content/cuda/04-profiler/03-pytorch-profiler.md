---
title: PyTorch Profiler
type: docs
math: true
sidebar:
  open: false
weight: 403
---

```python
import torch
import torch.nn as nn
import torch.profiler as profiler
from torch.profiler import profile, record_function


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def get_profiler_activities(
    device: str | torch.device | None = None, profile_cpu: bool = True
) -> list[profiler.ProfilerActivity]:
    if device is None:
        # Auto-detect best available accelerator, fallback to cpu
        device_type = (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "cpu"
        )
    else:
        device_type = torch.device(device).type

    activities = []

    if profile_cpu or device_type == "cpu":
        activities.append(profiler.ProfilerActivity.CPU)
    if device_type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    elif device_type == "xpu":
        activities.append(profiler.ProfilerActivity.XPU)

    if not activities:
        raise ValueError(
            f"Nothing to profile for device: {device}. Check your configuration."
        )

    return activities


device = "cuda" if torch.cuda.is_available() else "cpu"

# this includes total time spent including child operations
# use `self_{device}_time_total` to get self time only (excluding child operations)
sort_by_keyword = f"{device}_time_total" if device != "cpu" else "cpu_time_total"

model = SimpleModel().to(device)
inputs = torch.randn(4, 8).to(device)  # batch size of 4, input features of 8

with profile(
    activities=get_profiler_activities(device=device),
    record_shapes=False,  # adds cpu overhead if True, useful for debugging
) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
prof.export_chrome_trace("trace.json")
```

---

## Continuous Profiling with `Schedule`

When training a model over many steps, you don't want to profile everything, it creates massive files and slows down your training. Use a schedule to sample specific steps. Also, first few runs are often slower due to warmup, so you can skip them.

> [!IMPORTANT]
> **torch.profiler.profiler.schedule(*, wait, warmup, active, repeat=0, skip_first=0, skip_first_wait=0)** [source](https://docs.pytorch.org/docs/2.13/profiler.html#torch.profiler.profiler.schedule)
>
> Returns a callable that can be used as profiler schedule argument. The profiler will skip the first skip_first steps, then wait for wait steps, then do the warmup for the next warmup steps, then do the active recording for the next active steps and then repeat the cycle starting with wait steps. The optional number of cycles is specified with the repeat parameter, the zero value means that the cycles will continue until the profiling is finished.
> 
> The skip_first_wait parameter controls whether the first wait stage should be skipped. This can be useful if a user wants to wait longer than skip_first between cycles, but not for the first profile. For example, if skip_first is 10 and wait is 20, the first cycle will wait 10 + 20 = 30 steps before warmup if skip_first_wait is zero, but will wait only 10 steps if skip_first_wait is non-zero. All subsequent cycles will then wait 20 steps between the last active and warmup.

- `schedule` is simply a function that takes the current step and returns a `torch.profiler.ProfilerAction` enum value, which can be one of:
  - `torch.profiler.ProfilerAction.None`: no recording
  - `torch.profiler.ProfilerAction.WARMUP`: recording but not saving results
  - `torch.profiler.ProfilerAction.RECORD`: recording and storing for later
  - `torch.profiler.ProfilerAction.RECORD_AND_SAVE`: record & collect previous records & save to disk or do something based if user specified a `on_trace_ready` callback
 
```python
from torch.profiler import schedule, ProfilerAction

sch = schedule(wait=10, warmup=2, active=3, repeat=2, skip_first=5, skip_first_wait=0)

def expected_action(step: int) -> ProfilerAction:
    # 0-4: None (skip first)
    # 5-14: None (wait)
    # 15-16: WARMUP
    # 17-18: RECORD
    # 19: RECORD_AND_SAVE
    # 20-29: None (wait)
    # 30-31: WARMUP
    # 32-33: RECORD
    # 34: RECORD_AND_SAVE
    # 35-99: None
    if (step in (19,34)):
        return ProfilerAction.RECORD_AND_SAVE
    if (step in (17,18,32,33)):
        return ProfilerAction.RECORD
    if (step in (15,16,30,31)):
        return ProfilerAction.WARMUP
    return ProfilerAction.NONE

for i in range(200):
    assert sch(i) == expected_action(i), f"step {i}: {sch(i)} != {expected_action(i)}"

print("All assertions passed. The schedule behaves as expected. ✅")
```

## Usage:

```python
# Setup a schedule: Skip 2 steps, Warmup for 2 steps, Active profiling for 3 steps, Repeat.
my_schedule = schedule(
    wait=2,     # Steps to ignore at the start
    warmup=2,   # Steps to run to warm up the engines (not recorded)
    active=3,   # Steps where data is actually collected
    repeat=1    # Number of times to repeat this cycle (0 means forever)
)

def trace_handler(prof):
    # This function runs every time an 'active' cycle finishes
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    # Or save it for TensorBoard:
    # prof.export_chrome_trace(f"trace_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU],
    schedule=my_schedule,
    on_trace_ready=trace_handler  # Action to take when data is ready
) as prof:
    for step in range(10):
        # Your training loop code here
        model(inputs)
        prof.step()  # CRITICAL: You must call prof.step() at the end of every loop step!
```

- `prof.step()` internally checks the current step against the schedule and decides whether to record or not. It is essential to call this at the end of each iteration in your training loop to ensure the profiler behaves as expected.

---

## `record_function` Context Manager

The `record_function` context manager allows you to **label specific blocks of code** so they show up with custom names inside your profiling reports or Chrome/TensorBoard traces.

- By default, the PyTorch Profiler only records raw, low-level operator names (like `aten::conv2d` or `aten::add`). `record_function` bridges the gap between your high-level Python logic and those low-level operations.

- Wrap any block of PyTorch code—like a forward pass, data loading, or a loss calculation—using `with record_function("your_custom_label"):`.

* **CPU-Side Only Execution:** The context manager itself executes on the CPU to mark time boundaries. However, because PyTorch tracks which GPU/XPU kernels are launched during those boundaries, any accelerator activity triggered inside the block is accurately attributed to your custom label.
* **Negligible Overhead:** It is incredibly lightweight. You can use it generously across different phases of your training loop without worrying about distorting your performance metrics.

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    
    # Label the data loading step
    with record_function("🛑 DATA_LOADING"):
        inputs = torch.randn(5, 3, 224, 224)
        labels = torch.randint(0, 1000, (5,))

    # Label the model execution step
    with record_function("🚀 MODEL_FORWARD"):
        outputs = model(inputs)
        
    # Label the backward pass step
    with record_function("📉 BACKWARD_PASS"):
        loss = criterion(outputs, labels)
        loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

```

- shows up in table:

| Name | CPU Time | CUDA Time |
| --- | --- | --- |
| **🚀 MODEL_FORWARD** | **12.4ms** | **8.1ms** |
| ├── aten::convolution | 2.1ms | 1.8ms |
| └── aten::relu | 0.4ms | 0.2ms |
| **📉 BACKWARD_PASS** | **24.1ms** | **18.5ms** |

- shows up in `Chrome Traces`

![record function in Chrome Traces](/04-profiler/record-function.png)

---

## Some additional info

- **Overhead Warning**: Profiling adds stress to your CPU and memory. Turning on `with_stack=True` or `profile_memory=True` makes Python do a lot of extra bookkeeping, which can slightly slow down your overall run. Use them to diagnose bugs, then turn them off for raw speed tests.

- **The PyTorch Dispatcher**: When you run an operation like `torch.add()`, PyTorch passes it through a wrapper layer. If the profiler is active, this wrapper registers callbacks that save timestamps right before and after the operation runs.
