"""
Example experiment config that co-launches policy servers with each job.

Usage:
    python scripts/batch_eval.py --config experiments/with_servers.py
    python scripts/batch_eval.py --config experiments/with_servers.py --dry-run
"""

from polaris.config import EvalArgs, PolicyArgs, BatchConfig, PolicyServer, JobCfg


'''
Define reusable servers. Servers MUST accept a port argument in the command to 
avoid policy servers conflicting. `{port}` is auto-replaced with a free port
determined at runtime.
'''
PI0_FAST_SERVER = PolicyServer(
    name="pi0_fast",
    command=" ".join([
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35",
        "~/projects/PolaRiS/third_party/openpi/.venv/bin/python",
        "~/projects/PolaRiS/third_party/openpi/scripts/serve_policy.py",
        "--port {port}",
        "policy:checkpoint --policy.config pi0_fast_droid_jointpos",
        "--policy.dir gs://openpi-assets-simeval/pi0_fast_droid_jointpos",
    ]),
    ready_message="server listening on",
)

PI05_SERVER = PolicyServer(
    name="pi05",
    command=" ".join([
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.35",
        "~/projects/PolaRiS/third_party/openpi/.venv/bin/python",
        "~/projects/PolaRiS/third_party/openpi/scripts/serve_policy.py",
        "--port {port}",
        "policy:checkpoint --policy.config pi05_droid_jointpos",
        "--policy.dir gs://openpi-assets-simeval/pi05_droid_jointpos",
    ]),
    ready_message="server listening on",
)

# Each job references its server. The server will be launched on the same GPU as the job.
config = BatchConfig(
    jobs=[
    #     # pi0 jobs
    #     JobCfg(
    #         server=PI0_FAST_SERVER,
    #         eval_args=EvalArgs(
    #             environment="DROID-BlockStackKitchen",
    #             policy=PolicyArgs(
    #                 name="pi0_fast_droid_jointpos",
    #                 client="DroidJointPos",
    #                 open_loop_horizon=8,
    #             ),
    #         ),
    #     ),
    #     JobCfg(
    #         server=PI0_FAST_SERVER,
    #         eval_args=EvalArgs(
    #             environment="DROID-FoodBussing",
    #             policy=PolicyArgs(
    #                 name="pi0_fast_droid_jointpos",
    #                 client="DroidJointPos",
    #                 open_loop_horizon=8,
    #             ),
    #         ),
    #     ),
    #     JobCfg(
    #         server=PI0_FAST_SERVER,
    #         eval_args=EvalArgs(
    #             environment="DROID-PanClean",
    #             policy=PolicyArgs(
    #                 name="pi0_fast_droid_jointpos",
    #                 client="DroidJointPos",
    #                 open_loop_horizon=8,
    #             ),
    #         )
    #     ),

        # pi05 jobs
        JobCfg(
            server=PI05_SERVER,
            eval_args=EvalArgs(
                environment="DROID-BlockStackKitchen",
                policy=PolicyArgs(
                    name="pi05_droid_jointpos",
                    client="DroidJointPos",
                    open_loop_horizon=8,
                ),
            ),
        ),
        JobCfg(
            server=PI05_SERVER,
            eval_args=EvalArgs(
                environment="DROID-FoodBussing",
                policy=PolicyArgs(
                    name="pi05_droid_jointpos",
                    client="DroidJointPos",
                    open_loop_horizon=8,
                ),
            ),
        ),
        JobCfg(
            server=PI05_SERVER,
            eval_args=EvalArgs(
                environment="DROID-FoodBussing",
                policy=PolicyArgs(
                    name="pi05_droid_jointpos",
                    client="DroidJointPos",
                    open_loop_horizon=8,
                ),
            )
        ),
    ],
)
