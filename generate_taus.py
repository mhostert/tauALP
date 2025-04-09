import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def run_simulation(num_events, base_filename, beam_energy):
    # Execute the C++ program with the base filename and the number of events as command line arguments
    command = ["./ntaus", str(num_events), base_filename, str(beam_energy)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the result status
    if result.returncode != 0:
        print(f"Error with {base_filename}: {result.stderr.decode()}")
    else:
        print(f"Completed: {base_filename} with {num_events} events")


N_EVENTS = 100_000


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run tau simulations.")
    parser.add_argument(
        "--beam_energy",
        type=float,
        required=True,
        help="Beam energy for the simulation",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="name for this simulation to be used in saving files, e.g., NuMI_120GeV",
    )
    args = parser.parse_args()

    # Configuration for different runs
    tasks = [
        (N_EVENTS, "0"),
        (N_EVENTS, "1"),
        (N_EVENTS, "2"),
        (N_EVENTS, "3"),
        (N_EVENTS, "4"),
        (N_EVENTS, "5"),
        (N_EVENTS, "6"),
        (N_EVENTS, "7"),
    ]

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run_simulation,
                task[0],
                args.name + "_" + task[1],
                str(args.beam_energy),
            )
            for task in tasks
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                # Result will be None as the task doesn't return anything
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")


if __name__ == "__main__":
    main()
